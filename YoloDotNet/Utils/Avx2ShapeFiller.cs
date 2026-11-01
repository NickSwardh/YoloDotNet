// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Utils
{
    /// <summary>
    /// Fast drawing of filled shapes, such as segmentation masks and pose keypoints, using AVX2 on a single thread.
    /// The <c>Draw</c> methods and the pixel masks use it whenever the CPU supports AVX2.
    /// </summary>
    /// <remarks>
    /// <para><b>Words used in this file</b></para>
    /// <list type="bullet">
    /// <item><description><b>Outline (contour):</b> a closed polygon around the shape, as a list of points. A shape can have
    /// several, for example an object and a hole in it.</description></item>
    /// <item><description><b>Coverage:</b> how much of a pixel the shape covers, from 0 (outside) to 1 (fully inside). Edge pixels
    /// are partly covered, which is what makes edges look smooth.</description></item>
    /// <item><description><b>Cell:</b> one float in the working buffer, one per pixel of the area being filled.</description></item>
    /// <item><description><b>Span:</b> the cells of a row that the outline actually touched. Only those are cleared and processed.</description></item>
    /// <item><description><b>Strip:</b> a band of up to <see cref="STRIP_ROWS"/> rows, filled one at a time to keep memory small.</description></item>
    /// </list>
    /// <para><b>How a shape with smooth edges is filled</b></para>
    /// <list type="number">
    /// <item><description>Find the shape's bounding box, clipped to the image. Only that area is worked on.</description></item>
    /// <item><description>Every edge of every outline is walked row by row. In each row it passes through, it records how much of
    /// each cell lies to its right, weighted by how much of the row's height the edge covers. Edges going down add,
    /// edges going up subtract.</description></item>
    /// <item><description>Adding the cells of a row up from left to right then gives each pixel's coverage. Outside the shape the
    /// left and right edges cancel out to 0; inside they add up to 1.</description></item>
    /// <item><description>The coverage becomes an alpha value (0-255), and the shape's colour is blended over the image with that
    /// alpha, 8 pixels at a time with AVX2. For pixel masks the alpha value itself is written.</description></item>
    /// </list>
    /// A pixel on a shape's edge gets exactly the share of its area that lies inside the shape; a pixel fully inside gets
    /// the full colour. Holes stay empty: where outlines overlap they cancel out (the even-odd rule).
    /// <para>
    /// Sharp-edged masks (every pixel 0 or 255) take a simpler route: for each row, the pixels between pairs of outline
    /// crossings are set, so a pixel is inside exactly when its centre is.
    /// </para>
    /// <para>
    /// Large shapes are filled in strips of <see cref="STRIP_ROWS"/> rows, so the working memory stays small (about
    /// 1 MB for a shape as wide as a 4K image). Nothing is allocated per shape: buffers are reused per thread or borrowed
    /// from the shared pool and returned, so drawing from several threads at once is safe.
    /// </para>
    /// <para>
    /// Every <c>Try</c> method returns <see langword="false"/>, without changing anything, when it cannot handle the
    /// input: no AVX2 (older CPUs, ARM), an image that is not 32-bit RGBA/BGRA, or outline points that are not finite
    /// numbers. The caller then draws with SkiaSharp instead, so the output is always correct.
    /// </para>
    /// </remarks>
    internal static unsafe class Avx2ShapeFiller
    {
        // Rows filled at a time. Each strip needs one row of the working buffer per row.
        private const int STRIP_ROWS = 64;

        // Working buffers up to this many cells (64 KB) are kept per thread; larger ones are borrowed from the shared pool
        private const int SMALL_CELLS = 1 << 14;

        // Largest row of alpha values kept on the stack (8 KB, an 8K-wide image)
        private const int MAX_STACK_ALPHA = 8192 + 16;

        // A circle is filled as a polygon with the circle's area whose outline stays within about this many pixels of it
        private const float CIRCLE_TOLERANCE = 0.1f;

        // Working buffer for small shapes, one per thread so threads never share it. Created on first use, then reused.
        [ThreadStatic] private static float[]? _smallCoverage;

        // Circle outlines by number of corners, one set per thread. Cos and Sin hold the corner directions (already scaled
        // so the polygon has the circle's area); Polygon is reused for the actual corner points of each circle drawn.
        [ThreadStatic] private static Dictionary<int, (float[] Cos, float[] Sin, SKPoint[][] Polygon)>? _circles;

        /// <summary>
        /// True when this CPU supports AVX2, so the fast path can be used.
        /// </summary>
        public static bool IsSupported => Avx2.IsSupported;

        /// <summary>
        /// Fills a shape made of one or more closed outlines, such as a segmentation mask, with smooth edges.
        /// Holes in the shape stay empty.
        /// </summary>
        /// <param name="image">The image to draw on (32-bit RGBA or BGRA).</param>
        /// <param name="contours">The shape's outlines in image coordinates, for example <see cref="Segmentation.Contours"/>.</param>
        /// <param name="color">Fill colour; its alpha sets the opacity.</param>
        /// <returns>
        /// <see langword="true"/> when the shape was drawn (or lies outside the image); <see langword="false"/> when this
        /// input can't be handled here and nothing was changed. Draw it with SkiaSharp instead.
        /// </returns>
        public static bool TryFillPolygons(SKBitmap image, SKPoint[][] contours, SKColor color)
        {
            if (!IsSupported)
                return false;

            // One read of the image's description: every SKBitmap property is a call into SkiaSharp's native code
            var info = image.Info;

            // Only 32-bit images with premultiplied or opaque alpha are blended here; anything else goes back to the caller

            if (info.ColorType is not (SKColorType.Bgra8888 or SKColorType.Rgba8888) || info.AlphaType is not (SKAlphaType.Premul or SKAlphaType.Opaque))
                return false;

            var pixels = (byte*)image.GetPixels();

            if (pixels is null)
                return false;

            // The outlines are in image coordinates, so the area to fill starts at (0, 0) of the image
            var target = new Target(pixels, image.RowBytes, info.Width, info.Height, blend: true, info.ColorType == SKColorType.Bgra8888, color);
            return Fill(contours, 0, 0, target);
        }

        /// <summary>
        /// Fills a circle, such as a pose keypoint, with smooth edges.
        /// </summary>
        /// <param name="image">The image to draw on (32-bit RGBA or BGRA).</param>
        /// <param name="center">Centre of the circle in image coordinates.</param>
        /// <param name="radius">Radius in pixels. Any size works: larger circles get more points on their outline.</param>
        /// <param name="color">Fill colour; its alpha sets the opacity.</param>
        /// <returns>
        /// <see langword="true"/> when the circle was drawn (or lies outside the image); <see langword="false"/> when this
        /// input can't be handled here and nothing was changed. Draw it with SkiaSharp instead.
        /// </returns>
        public static bool TryFillCircle(SKBitmap image, SKPoint center, float radius, SKColor color)
        {
            if (!IsSupported || !float.IsFinite(radius))
                return false;

            if (radius <= 0)
                return true;

            // Enough points that each side stays within the tolerance of the circle
            int segments = Math.Max(8, (int)MathF.Ceiling(MathF.PI / MathF.Acos(1 - MathF.Min(CIRCLE_TOLERANCE / radius, 1))));
            var cache = _circles ??= [];

            if (!cache.TryGetValue(segments, out var circle))
            {
                var cos = new float[segments];
                var sin = new float[segments];

                // Corners slightly outside the circle, so the polygon has exactly the circle's area. Corners on the circle
                // would cut off a little area along every side, which makes small circles visibly smaller.
                double step = 2 * Math.PI / segments;
                float scale = (float)Math.Sqrt(step / Math.Sin(step));

                for (int i = 0; i < segments; i++)
                {
                    cos[i] = scale * MathF.Cos((float)(step * i));
                    sin[i] = scale * MathF.Sin((float)(step * i));
                }

                cache[segments] = circle = (cos, sin, [new SKPoint[segments]]);
            }

            // Place the corners around this circle's centre, then fill the polygon like any other shape
            var points = circle.Polygon[0];

            for (int i = 0; i < segments; i++)
                points[i] = new SKPoint(center.X + radius * circle.Cos[i], center.Y + radius * circle.Sin[i]);

            return TryFillPolygons(image, circle.Polygon, color);
        }

        /// <summary>
        /// Writes a shape as a mask: one byte per pixel, 255 inside the shape and 0 outside. Used for
        /// <see cref="ImageExtension.GetPixelMask(Segmentation, bool)"/> and <see cref="Segmentation.BitPackedPixelMask"/>.
        /// </summary>
        /// <param name="pixels">The mask to write into, one byte per pixel. It must start out all 0: only pixels the shape touches are written.</param>
        /// <param name="width">Width of the mask in pixels.</param>
        /// <param name="height">Height of the mask in pixels.</param>
        /// <param name="rowBytes">Bytes from the start of one mask row to the next.</param>
        /// <param name="contours">The shape's outlines, for example <see cref="Segmentation.Contours"/>.</param>
        /// <param name="left">Where the mask's left edge is, in the outlines' coordinates.</param>
        /// <param name="top">Where the mask's top edge is, in the outlines' coordinates.</param>
        /// <param name="antiAlias">
        /// <see langword="true"/> for smooth edges: edge pixels get a value between 0 and 255 for how much of them is inside,
        /// exactly as the drawn mask. <see langword="false"/> for sharp edges: a pixel is 255 when its centre is inside the shape.
        /// </param>
        /// <returns><see langword="false"/>, without writing anything, when the CPU has no AVX2 or a point is not finite. Use SkiaSharp instead.</returns>
        public static bool TryRasterize(byte* pixels, int width, int height, int rowBytes, SKPoint[][] contours, float left, float top, bool antiAlias)
        {
            if (!IsSupported || pixels is null)
                return false;

            // Sharp edges: a simpler scanline fill is enough, every pixel ends up 0 or 255
            if (!antiAlias)
                return FillSharp(pixels, width, height, rowBytes, contours, left, top);

            // Smooth edges: the same fill as drawing, but each pixel's alpha value (0-255) is written instead of blended
            var target = new Target(pixels, rowBytes, width, height, blend: false, bgra: false, SKColors.White);
            return Fill(contours, left, top, target);
        }

        /// <summary>
        /// Where a fill goes: blended over 32-bit pixels in <see cref="Color"/>, or written as one byte per pixel.
        /// </summary>
        private readonly struct Target(byte* pixels, int rowBytes, int width, int height, bool blend, bool bgra, SKColor color)
        {
            /// <summary>First byte of the image or mask.</summary>
            public readonly byte* Pixels = pixels;

            /// <summary>Bytes from the start of one row to the next (can be more than the visible width).</summary>
            public readonly int RowBytes = rowBytes;

            /// <summary>Width in pixels; nothing is written right of it.</summary>
            public readonly int Width = width;

            /// <summary>Height in pixels; nothing is written below it.</summary>
            public readonly int Height = height;

            /// <summary>True: blend <see cref="Color"/> over 32-bit pixels. False: write alpha values into a one-byte-per-pixel mask.</summary>
            public readonly bool Blend = blend;

            /// <summary>True when the image stores blue first (BGRA), false for red first (RGBA).</summary>
            public readonly bool Bgra = bgra;

            /// <summary>The fill colour; its alpha is the opacity of a fully covered pixel.</summary>
            public readonly SKColor Color = color;
        }

        /// <summary>
        /// Fills the outlines with smooth edges into <paramref name="target"/>: finds the area to work on, prepares the
        /// buffers, then handles that area one strip of rows at a time.
        /// </summary>
        /// <param name="contours">The outlines.</param>
        /// <param name="originX">Where the target's left edge is, in the outlines' coordinates (0 for an image).</param>
        /// <param name="originY">Where the target's top edge is, in the outlines' coordinates (0 for an image).</param>
        /// <param name="target">Where the result goes.</param>
        /// <returns><see langword="false"/> when a point is not a finite number, without writing anything.</returns>
        private static bool Fill(SKPoint[][] contours, float originX, float originY, Target target)
        {
            // A fully transparent colour changes nothing
            if (target.Blend && target.Color.Alpha == 0)
                return true;

            // Bounding box of the outlines. A point that is not a finite number goes to SkiaSharp.
            float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;

            foreach (var contour in contours)
            {
                foreach (var point in contour)
                {
                    if (!float.IsFinite(point.X) || !float.IsFinite(point.Y))
                        return false;

                    if (point.X < minX) minX = point.X;
                    if (point.X > maxX) maxX = point.X;
                    if (point.Y < minY) minY = point.Y;
                    if (point.Y > maxY) maxY = point.Y;
                }
            }

            if (minX > maxX)
                return true;

            // The box in target pixels, clipped to the target before converting to whole numbers
            int left = (int)Math.Clamp(MathF.Floor(minX - originX), 0, target.Width);
            int top = (int)Math.Clamp(MathF.Floor(minY - originY), 0, target.Height);
            int right = (int)Math.Clamp(MathF.Ceiling(maxX - originX), 0, target.Width);
            int bottom = (int)Math.Clamp(MathF.Ceiling(maxY - originY), 0, target.Height);
            int width = right - left, height = bottom - top;

            if (width <= 0 || height <= 0)
                return true;

            // The working buffer has one row of cells per pixel row of a strip. Rows are padded to whole 8-cell blocks (so AVX2
            // can always read 8 cells at a time), with room for the cell just past the right edge that an edge may write to.
            int stride = (width + 2 + 7) & ~7;
            int stripRows = Math.Min(height, STRIP_ROWS);
            int cells = stride * stripRows;
            bool small = cells <= SMALL_CELLS;

            float[] coverageArray = small ? (_smallCoverage ??= new float[SMALL_CELLS]) : ArrayPool<float>.Shared.Rent(cells);

            // For each row of a strip, the first and last cell its edges touched (see Touch)
            int* spans = stackalloc int[STRIP_ROWS * 2];

            // One row of alpha values: on the stack for images up to 8K wide, borrowed from the pool beyond that
            byte[]? alphaArray = stride + 16 > MAX_STACK_ALPHA ? ArrayPool<byte>.Shared.Rent(stride + 16) : null;
            byte* stackAlpha = stackalloc byte[alphaArray is null ? stride + 16 : 0];

            try
            {
                fixed (float* coverage = coverageArray)
                fixed (byte* pooledAlpha = alphaArray)
                {
                    byte* alphaRow = alphaArray is null ? stackAlpha : pooledAlpha;

                    for (int stripTop = 0; stripTop < height; stripTop += STRIP_ROWS)
                    {
                        int rows = Math.Min(STRIP_ROWS, height - stripTop);

                        // Mark every row of the strip as untouched: no cells cleared or used yet
                        for (int y = 0; y < rows; y++)
                        {
                            spans[2 * y] = int.MaxValue;
                            spans[2 * y + 1] = -1;
                        }

                        // Outline coordinates relative to the strip's top-left pixel
                        float offsetX = originX + left, offsetY = originY + top + stripTop;

                        // Record every edge in the strip's rows, then turn the rows into pixels
                        AddAreas(contours, offsetX, offsetY, coverage, spans, stride, width, rows);

                        WriteRows(coverage, spans, alphaRow, stride, width, rows, left, top + stripTop, target);
                    }
                }
            }
            finally
            {
                // Borrowed buffers always go back to the pool, also if something above throws
                if (alphaArray is not null)
                    ArrayPool<byte>.Shared.Return(alphaArray);

                if (!small)
                    ArrayPool<float>.Shared.Return(coverageArray);
            }

            return true;
        }

        /// <summary>
        /// Adds every edge's area to the strip's rows (smooth edges). Coordinates are relative to the strip's top-left pixel.
        /// </summary>
        private static void AddAreas(SKPoint[][] contours, float offsetX, float offsetY, float* coverage, int* spans, int stride, int width, int rows)
        {
            foreach (var contour in contours)
            {
                for (int i = 0, j = contour.Length - 1; i < contour.Length; j = i++)
                {
                    float y0 = contour[j].Y - offsetY, y1 = contour[i].Y - offsetY;

                    // Edges entirely above or below this strip add nothing to it
                    if ((y0 <= 0 && y1 <= 0) || (y0 >= rows && y1 >= rows))
                        continue;

                    AddEdgeArea(coverage, spans, stride, width, rows, contour[j].X - offsetX, y0, contour[i].X - offsetX, y1);
                }
            }
        }

        /// <summary>
        /// Writes a sharp-edged mask: a pixel is 255 when its centre is inside the shape. For every row, the places where
        /// the outline crosses the line through the pixel centres are collected and put in order; the pixels between the
        /// 1st and 2nd crossing, the 3rd and 4th, and so on are inside (the even-odd rule).
        /// </summary>
        private static bool FillSharp(byte* pixels, int width, int height, int rowBytes, SKPoint[][] contours, float left, float top)
        {
            if (width <= 0 || height <= 0)
                return true;

            // Step 1: count the crossings per row. rowStart[y + 1] counts row y; summing them up afterwards turns rowStart[y]
            // into where row y's crossings begin in one shared list, so no list per row is needed.
            int[] rowStart = ArrayPool<int>.Shared.Rent(height + 1);
            int[]? cells = null;

            try
            {
                Array.Clear(rowStart, 0, height + 1);

                foreach (var contour in contours)
                {
                    for (int i = 0, j = contour.Length - 1; i < contour.Length; j = i++)
                    {
                        if (!float.IsFinite(contour[i].X) || !float.IsFinite(contour[i].Y))
                            return false;

                        var (yStart, yEnd) = CentreRows(contour[j].Y - top, contour[i].Y - top, height);

                        for (int y = yStart; y < yEnd; y++)
                            rowStart[y + 1]++;
                    }
                }

                for (int y = 0; y < height; y++)
                    rowStart[y + 1] += rowStart[y];

                int count = rowStart[height];

                if (count == 0)
                    return true;

                // Step 2: compute every crossing and put it in its row's part of the list. A crossing is stored as the first pixel
                // whose centre is right of it. 'next' says where the next crossing of each row goes.
                cells = ArrayPool<int>.Shared.Rent(count);
                int[] next = ArrayPool<int>.Shared.Rent(height);

                try
                {
                    Array.Copy(rowStart, next, height);

                    foreach (var contour in contours)
                    {
                        for (int i = 0, j = contour.Length - 1; i < contour.Length; j = i++)
                        {
                            float x0 = contour[j].X - left, y0 = contour[j].Y - top;
                            float x1 = contour[i].X - left, y1 = contour[i].Y - top;
                            var (yStart, yEnd) = CentreRows(y0, y1, height);

                            if (yStart >= yEnd)
                                continue;

                            float dxdy = (x1 - x0) / (y1 - y0);

                            for (int y = yStart; y < yEnd; y++)
                            {
                                // A crossing outside the mask is moved to its edge, where it still marks the rest of the row
                                float x = x0 + (y + 0.5f - y0) * dxdy;
                                cells[next[y]++] = (int)Math.Clamp(MathF.Floor(x + 0.5f), 0, width);
                            }
                        }
                    }
                }
                finally
                {
                    ArrayPool<int>.Shared.Return(next);
                }

                // Step 3: per row, put the crossings in order from left to right and fill between each pair
                for (int y = 0; y < height; y++)
                {
                    int from = rowStart[y], to = rowStart[y + 1];

                    // A row holds only a few crossings, so a simple insertion sort puts them in order
                    for (int k = from + 1; k < to; k++)
                    {
                        int cell = cells[k], m = k - 1;

                        while (m >= from && cells[m] > cell)
                        {
                            cells[m + 1] = cells[m];
                            m--;
                        }

                        cells[m + 1] = cell;
                    }

                    // A closed outline crosses every row an even number of times; an odd count from rounding drops the last
                    byte* row = pixels + (long)y * rowBytes;

                    for (int k = from; k + 1 < to; k += 2)
                        new Span<byte>(row + cells[k], cells[k + 1] - cells[k]).Fill(255);
                }
            }
            finally
            {
                if (cells is not null)
                    ArrayPool<int>.Shared.Return(cells);

                ArrayPool<int>.Shared.Return(rowStart);
            }

            return true;
        }

        /// <summary>
        /// The rows whose pixel centre lies between y0 and y1 (top included, bottom not), within [0, height).
        /// For example, an edge from y = 2.3 to y = 5.6 crosses the centres of rows 2 (at 2.5), 3, 4 and 5 (at 5.5): rows 2 to 6, end excluded.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static (int Start, int End) CentreRows(float y0, float y1, int height)
        {
            if (y0 > y1)
                (y0, y1) = (y1, y0);

            return ((int)Math.Clamp(MathF.Floor(y0 + 0.5f), 0, height), (int)Math.Clamp(MathF.Floor(y1 + 0.5f), 0, height));
        }

        /// <summary>
        /// Adds an edge's area to the rows it crosses. A part of the edge left of the box (x &lt; 0) is moved onto the
        /// left border as a vertical edge, and a part right of it onto the right border: for every pixel inside the box
        /// that covers exactly the same area.
        /// </summary>
        /// <remarks>
        /// Why moving is enough: a pixel inside the box only cares how much of the edge lies to its left, not exactly
        /// where. Everything left of the box is left of every pixel in it, so it may as well sit on the left border.
        /// Edges of shapes inside the image never leave the box, so this is the common, fast case.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void AddEdgeArea(float* coverage, int* spans, int stride, int width, int rows, float x0, float y0, float x1, float y1)
        {
            if (x0 >= 0 && x0 <= width && x1 >= 0 && x1 <= width)
                AddArea(coverage, spans, stride, width, rows, x0, y0, x1, y1);
            else
                AddSplitEdgeArea(coverage, spans, stride, width, rows, x0, y0, x1, y1);
        }

        /// <summary>
        /// The rare case of <see cref="AddEdgeArea"/>: an edge that reaches outside the box on the left or right.
        /// </summary>
        private static void AddSplitEdgeArea(float* coverage, int* spans, int stride, int width, int rows, float x0, float y0, float x1, float y1)
        {
            // Split the edge where it crosses x = 0 and x = width
            Span<float> t = stackalloc float[4];
            int count = 0;
            t[count++] = 0;

            if (x0 != x1)
            {
                float tLeft = -x0 / (x1 - x0), tRight = (width - x0) / (x1 - x0);

                if (tLeft > 0 && tLeft < 1) t[count++] = tLeft;
                if (tRight > 0 && tRight < 1) t[count++] = tRight;
            }

            t[count++] = 1;
            t[..count].Sort();

            for (int k = 0; k + 1 < count; k++)
            {
                float xa = x0 + (x1 - x0) * t[k], ya = y0 + (y1 - y0) * t[k];
                float xb = x0 + (x1 - x0) * t[k + 1], yb = y0 + (y1 - y0) * t[k + 1];
                AddArea(coverage, spans, stride, width, rows, Math.Clamp(xa, 0, width), ya, Math.Clamp(xb, 0, width), yb);
            }
        }

        /// <summary>
        /// For each row an edge crosses, adds how much of each cell lies to the right of the edge, weighted by the part of
        /// the row's height the edge spans (negative for edges going up). Each cell gets the change from the cell before,
        /// so that adding the cells up from left to right later gives the coverage.
        /// </summary>
        /// <remarks>
        /// Think of one row as a line of cells. An edge going down through the row says: everything to my right is inside,
        /// by the amount of the row's height I cover. A cell fully right of the edge gets that whole amount; a cell the
        /// edge passes through gets only the part of it right of the edge. Storing the change from cell to cell (instead
        /// of the amounts themselves) means only the few cells near the edge are written; adding up the row later
        /// restores the amounts.
        /// <para>
        /// Within one row the edge runs straight from xa to xb, so its x positions are spread evenly over that range. The
        /// share of cell c to the right of the edge is then the average of clamp(c + 1 - x, 0, 1) over x in [xa, xb]:
        /// (R(c + 1 - xa) - R(c + 1 - xb)) / (xb - xa), where R(u), the running total of clamp(u, 0, 1), is 0 for u &lt;= 0,
        /// u² / 2 for u up to 1, and u - 1/2 beyond.
        /// </para>
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void AddArea(float* coverage, int* spans, int stride, int width, int rows, float x0, float y0, float x1, float y1)
        {
            if (y0 == y1)
                return;

            // Walk every edge from top to bottom. An edge that really goes up subtracts instead of adds, so that the left
            // and right sides of a shape cancel out outside it.
            float direction = 1;

            if (y0 > y1)
            {
                direction = -1;
                (x0, x1) = (x1, x0);
                (y0, y1) = (y1, y0);
            }

            float dxdy = (x1 - x0) / (y1 - y0);
            int yStart = (int)MathF.Max(0, MathF.Floor(y0));
            int yEnd = (int)MathF.Min(rows, MathF.Ceiling(y1));

            // Where the edge enters the first row; each row then starts where the previous one ended
            float xTop = x0 + (MathF.Max(yStart, y0) - y0) * dxdy;

            for (int y = yStart; y < yEnd; y++)
            {
                // Plain comparisons instead of MathF.Min/Max: all values are finite, so their extra NaN handling isn't needed
                float yTop = y > y0 ? y : y0, yBottom = y + 1 < y1 ? y + 1 : y1;
                float xBottom = xTop + (yBottom - yTop) * dxdy;
                float height = (yBottom - yTop) * direction;

                // Where the edge enters and leaves this row, kept inside the box against rounding
                float xa = InBox(xTop < xBottom ? xTop : xBottom, width);
                float xb = InBox(xTop < xBottom ? xBottom : xTop, width);
                xTop = xBottom;

                if (height == 0)
                    continue;

                int first = (int)xa;
                float* row = coverage + y * stride;

                if (xb <= first + 1)
                {
                    // The edge stays within one cell: on average, the part of that cell right of the edge is
                    // first + 1 - (xa + xb) / 2. The next cell, and every cell after it, lies fully to the right.
                    float rightOfEdge = first + 1 - 0.5f * (xa + xb);

                    Touch(row, spans + 2 * y, first & ~7, (first + 1) | 7);
                    row[first] += height * rightOfEdge;
                    row[first + 1] += height * (1 - rightOfEdge);
                    continue;
                }

                // The edge crosses several cells. Cell 'last' is the first one fully right of the edge.
                int last = (int)MathF.Ceiling(xb);
                float perWidth = height / (xb - xa);

                Touch(row, spans + 2 * y, first & ~7, last | 7);

                // The remarks' formula, split by where each cell is. Each cell gets the change in area from the cell before.
                // The cell where the edge starts: a parabola, R(first + 1 - xa) = (first + 1 - xa)² / 2
                float startPart = first + 1 - xa;
                float previous = perWidth * 0.5f * startPart * startPart;
                row[first] += previous;

                // Cells the edge crosses from side to side: a straight line, (cell + 1/2 - xa) per unit of width
                int end = last - 1;

                for (int cell = first + 1; cell < end; cell++)
                {
                    float area = perWidth * (cell + 0.5f - xa);
                    row[cell] += area - previous;
                    previous = area;
                }

                // The cell where the edge ends: the same line, less the parabola past xb, R(end + 1 - xb)
                float endPart = last - xb;
                float endArea = perWidth * (end + 0.5f - xa - 0.5f * endPart * endPart);
                row[end] += endArea - previous;

                // Cell 'last' and every cell after it lie fully right of the edge
                row[last] += height - endArea;
            }
        }

        /// <summary>
        /// Keeps an x position within [0, width].
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float InBox(float x, int width)
            => x < 0 ? 0 : x > width ? width : x;

        /// <summary>
        /// Widens a row's used span to [from, to] (whole 8-cell blocks), clearing only cells not cleared before. Rows are
        /// cleared this way, on first use, instead of clearing the whole buffer for every shape.
        /// </summary>
        /// <remarks>
        /// The buffer is reused between shapes, so it still holds old values. Each row remembers the first and last cell
        /// in use (<c>span[0]</c> and <c>span[1]</c>, -1 when the row is untouched). Every cell inside that range has been
        /// cleared; cells outside it are never read.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Touch(float* row, int* span, int from, int to)
        {
            int low = span[0], high = span[1];

            if (high < 0)
            {
                new Span<float>(row + from, to - from + 1).Clear();
                span[0] = from;
                span[1] = to;
                return;
            }

            if (from < low)
            {
                new Span<float>(row + from, low - from).Clear();
                span[0] = from;
            }

            if (to > high)
            {
                new Span<float>(row + high + 1, to - high).Clear();
                span[1] = to;
            }
        }

        /// <summary>
        /// Adds up each row from left to right to get the coverage, then blends it over, or writes it into, the target.
        /// </summary>
        /// <remarks>
        /// Per row, in two passes:
        /// <list type="number">
        /// <item><description>Coverage to alpha: a running total along the row's cells, 8 at a time, gives each pixel's
        /// coverage; the even-odd rule folds it to 0-1, and it becomes an alpha value 0-255 in <paramref name="alphaRow"/>.</description></item>
        /// <item><description>Alpha to pixels: 8 pixels at a time, new = colour × alpha + old × (255 - alpha), divided by 255.
        /// Groups of 8 pixels the shape doesn't touch are skipped.</description></item>
        /// </list>
        /// </remarks>
        private static void WriteRows(float* coverage, int* spans, byte* alphaRow, int stride, int width, int rows, int left, int top, Target target)
        {
            var color = target.Color;
            ushort c0 = target.Bgra ? color.Blue : color.Red, c1 = color.Green, c2 = target.Bgra ? color.Red : color.Blue;
            float alpha = target.Blend ? color.Alpha : 255;

            // Constants for the whole call. Pixels are handled as 16-bit numbers during the blend, so the 8-bit values can be
            // multiplied without overflowing (255 × 255 fits in 16 bits).
            // Colour per channel for 4 pixels; the alpha channel blends as colour 255, which keeps opaque pixels opaque
            var colorVector = Vector256.Create(c0, c1, c2, (ushort)255, c0, c1, c2, 255, c0, c1, c2, 255, c0, c1, c2, 255);
            var spread = Vector128.Create((byte)0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
            var v127 = Vector256.Create((ushort)127);
            var v255 = Vector256.Create((ushort)255);
            var div255 = Vector256.Create((ushort)0x8081);
            var alphaVector = Vector256.Create(alpha);
            var half = Vector256.Create(0.5f);
            var one = Vector256.Create(1f);
            var signMask = Vector256.Create(-0f);
            var lastLane = Vector256.Create(7);

            // Fully covered pixels: the shape's colour and the share of the old pixel are the same for every pixel
            byte fullAlpha = (byte)alpha;
            ulong full8 = 0x0101010101010101UL * fullAlpha;
            var sourceFull = Div255(Avx2.MultiplyLow(colorVector, Vector256.Create((ushort)fullAlpha)), v127, div255);
            var remainingFull = Vector256.Create((ushort)(255 - fullAlpha));

            // Lane masks for 1-7 pixels: 8 ints loaded from laneMasks + 8 - count start with count all-ones lanes
            int* laneMasks = stackalloc int[16] { -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0 };

            for (int y = 0; y < rows; y++)
            {
                int x0 = spans[2 * y], x1 = Math.Min(spans[2 * y + 1], width - 1);

                if (x1 < x0)
                    continue;

                float* row = coverage + y * stride;

                // Running total from the span's left edge (where it is 0), 8 cells at a time. The cells are cleared in
                // whole blocks of 8 (see Touch), so the last block can be read in full.
                var carry = Vector256<float>.Zero;

                for (int x = x0; x <= x1; x += 8)
                {
                    // Running total of 8 cells at once: add each cell to its right neighbour, then to the cell 2 further on,
                    // within each half of 4; then add the first half's total to the second half; then add the total of all
                    // previous blocks (carry). Afterwards carry holds this block's last total, for the next block.
                    var sum = Avx.LoadVector256(row + x);
                    sum = Avx.Add(sum, Avx2.ShiftLeftLogical128BitLane(sum.AsInt32(), 4).AsSingle());
                    sum = Avx.Add(sum, Avx2.ShiftLeftLogical128BitLane(sum.AsInt32(), 8).AsSingle());
                    sum = Avx.Add(sum, Avx.Permute(Avx2.Permute2x128(sum, sum, 0x08), 0xFF));
                    sum = Avx.Add(sum, carry);
                    carry = Avx2.PermuteVar8x32(sum, lastLane);

                    // Even-odd rule: the distance to the nearest even number is 0 outside, 1 inside, the coverage on an edge
                    var even = Avx.RoundToNearestInteger(Avx.Multiply(sum, half));
                    var inside = Avx.Min(Avx.AndNot(signMask, Avx.Subtract(sum, Avx.Add(even, even))), one);

                    // Coverage × alpha, rounded, packed down to 8 bytes and stored as this block's alpha values
                    var a = Avx.ConvertToVector256Int32WithTruncation(Avx.Add(Avx.Multiply(inside, alphaVector), half));
                    var a16 = Avx2.PackSignedSaturate(a, a);
                    var a8 = Avx2.PackUnsignedSaturate(a16, a16);
                    *(uint*)(alphaRow + x - x0) = a8.AsUInt32().GetElement(0);
                    *(uint*)(alphaRow + x - x0 + 4) = a8.AsUInt32().GetElement(4);
                }

                int n = x1 - x0 + 1;

                // Pixel masks: the alpha values are the result
                if (!target.Blend)
                {
                    Buffer.MemoryCopy(alphaRow, target.Pixels + (long)(top + y) * target.RowBytes + left + x0, n, n);
                    continue;
                }

                // Blend 8 pixels at a time, skipping pixels the shape doesn't cover. The last 1-7 pixels of a row are
                // read and written through a lane mask, so no pixel after the row's end is touched.
                byte* pixel = target.Pixels + (long)(top + y) * target.RowBytes + (left + x0) * 4;

                for (int i = 0; i < n; i += 8, pixel += 32)
                {
                    ulong alpha8 = *(ulong*)(alphaRow + i);

                    if (alpha8 == 0)
                        continue;

                    // Read the 8 pixels (or the last 1-7 of the row) and widen each byte to 16 bits for the arithmetic
                    int count = Math.Min(8, n - i);
                    var lanes = Avx.LoadVector256(laneMasks + 8 - count);
                    var destination = count == 8 ? Avx.LoadVector256((int*)pixel) : Avx2.MaskLoad((int*)pixel, lanes);

                    var dLow = Avx2.ConvertToVector256Int16(destination.GetLower().AsByte()).AsUInt16();
                    var dHigh = Avx2.ConvertToVector256Int16(destination.GetUpper().AsByte()).AsUInt16();
                    Vector256<ushort> oLow, oHigh;

                    // All 8 fully covered (common inside a shape): the colour part is the same for every pixel
                    if (alpha8 == full8)
                    {
                        oLow = Avx2.Add(sourceFull, Div255(Avx2.MultiplyLow(dLow, remainingFull), v127, div255));
                        oHigh = Avx2.Add(sourceFull, Div255(Avx2.MultiplyLow(dHigh, remainingFull), v127, div255));
                    }
                    else
                    {
                        // Each pixel's own alpha, repeated for its 4 channels
                        var aLow = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(Vector128.CreateScalarUnsafe((uint)alpha8).AsByte(), spread)).AsUInt16();
                        var aHigh = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(Vector128.CreateScalarUnsafe((uint)(alpha8 >> 32)).AsByte(), spread)).AsUInt16();

                        // New pixel = colour × alpha + old pixel × (255 − alpha), each part divided by 255 and rounded separately
                        oLow = Avx2.Add(Div255(Avx2.MultiplyLow(colorVector, aLow), v127, div255),
                            Div255(Avx2.MultiplyLow(dLow, Avx2.Subtract(v255, aLow)), v127, div255));
                        oHigh = Avx2.Add(Div255(Avx2.MultiplyLow(colorVector, aHigh), v127, div255),
                            Div255(Avx2.MultiplyLow(dHigh, Avx2.Subtract(v255, aHigh)), v127, div255));
                    }

                    // Packing to bytes works per 128-bit half (pixels 0-1, 4-5, 2-3, 6-7); reorder them to 0-7
                    var packed = Avx2.PackUnsignedSaturate(oLow.AsInt16(), oHigh.AsInt16());
                    var result = Avx2.Permute4x64(packed.AsUInt64(), 0b_11_01_10_00).AsInt32();

                    if (count == 8)
                        Avx.Store((int*)pixel, result);
                    else
                        Avx2.MaskStore((int*)pixel, lanes, result);
                }
            }
        }

        /// <summary>
        /// Divides every value by 255 with rounding, (v + 127) / 255, exact for values up to 255 × 255.
        /// </summary>
        /// <remarks>
        /// Dividing is slow, so it is done as a multiplication and a shift: (v + 127) × 32,897 / 2²³ gives the same whole
        /// number as (v + 127) / 255 for every value that can occur here. MultiplyHigh keeps the top 16 bits of the
        /// product (a shift by 16), and the final shift by 7 completes the 23.
        /// </remarks>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<ushort> Div255(Vector256<ushort> v, Vector256<ushort> v127, Vector256<ushort> div255)
            => Avx2.ShiftRightLogical(Avx2.MultiplyHigh(Avx2.Add(v, v127), div255), 7);
    }
}
