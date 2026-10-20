// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Utils
{
    /// <summary>
    /// Traces the outlines of a segmentation mask with marching squares.
    ///
    /// The mask is a continuous field sampled on a grid, padded with outside samples so every outline is a closed loop,
    /// including holes and separate parts. Edge crossings are interpolated linearly between samples, which places the
    /// outline with sub-sample precision.
    /// </summary>
    internal static class MaskContourTracer
    {
        // Private pool: every rented edge table is handed back with all entries reset to zero, so tables never
        // need clearing, which would otherwise cost time proportional to the mask area on every call.
        private static readonly ArrayPool<int> _edgeTablePool = ArrayPool<int>.Create(1 << 24, 16);

        /// <summary>
        /// Traces all outlines of a continuous field where it crosses <paramref name="iso"/>; samples above it are inside.
        /// Field sample (i, j) is mapped to (i * scaleX + offsetX, j * scaleY + offsetY).
        /// </summary>
        public static List<SKPoint[]> TraceField(ReadOnlySpan<float> field, int width, int height, float iso,
            float scaleX, float offsetX, float scaleY, float offsetY)
            => TraceCore(field, width, height, iso, outside: iso - 1f, scaleX, offsetX, scaleY, offsetY);

        /// <summary>
        /// Clips a closed polygon to a rectangle (Sutherland-Hodgman).
        /// </summary>
        public static SKPoint[] ClipToRectangle(SKPoint[] polygon, SKRect rectangle)
        {
            bool inside = true;

            foreach (var p in polygon)
            {
                if (p.X < rectangle.Left || p.X > rectangle.Right || p.Y < rectangle.Top || p.Y > rectangle.Bottom)
                {
                    inside = false;
                    break;
                }
            }

            if (inside)
                return polygon;

            var points = new List<SKPoint>(polygon);
            points = ClipEdge(points, p => p.X >= rectangle.Left, (a, b) => Intersect(a, b, (rectangle.Left - a.X) / (b.X - a.X)));
            points = ClipEdge(points, p => p.X <= rectangle.Right, (a, b) => Intersect(a, b, (rectangle.Right - a.X) / (b.X - a.X)));
            points = ClipEdge(points, p => p.Y >= rectangle.Top, (a, b) => Intersect(a, b, (rectangle.Top - a.Y) / (b.Y - a.Y)));
            points = ClipEdge(points, p => p.Y <= rectangle.Bottom, (a, b) => Intersect(a, b, (rectangle.Bottom - a.Y) / (b.Y - a.Y)));

            return [.. points];

            static SKPoint Intersect(SKPoint a, SKPoint b, float t) => new(a.X + (b.X - a.X) * t, a.Y + (b.Y - a.Y) * t);

            static List<SKPoint> ClipEdge(List<SKPoint> input, Func<SKPoint, bool> keep, Func<SKPoint, SKPoint, SKPoint> intersect)
            {
                var output = new List<SKPoint>(input.Count);

                for (int i = 0; i < input.Count; i++)
                {
                    var current = input[i];
                    var previous = input[(i + input.Count - 1) % input.Count];
                    bool currentIn = keep(current), previousIn = keep(previous);

                    if (currentIn)
                    {
                        if (!previousIn)
                            output.Add(intersect(previous, current));

                        output.Add(current);
                    }
                    else if (previousIn)
                    {
                        output.Add(intersect(previous, current));
                    }
                }

                return output;
            }
        }

        /// <summary>
        /// Simplifies a closed outline with the Ramer-Douglas-Peucker algorithm, keeping every point that deviates
        /// more than <paramref name="tolerance"/> pixels from the simplified outline.
        /// </summary>
        public static SKPoint[] Simplify(SKPoint[] contour, float tolerance)
        {
            if (contour.Length <= 4)
                return contour;

            // Split the loop at the point farthest from the first point, and simplify both halves.
            int far = 0;
            float farDistance = -1;

            for (int i = 1; i < contour.Length; i++)
            {
                float distance = SKPoint.DistanceSquared(contour[0], contour[i]);

                if (distance > farDistance)
                    (far, farDistance) = (i, distance);
            }

            // Index contour.Length stands for the first point again, closing the loop.
            var keep = new bool[contour.Length + 1];
            keep[0] = keep[far] = keep[contour.Length] = true;

            var stack = new Stack<(int Start, int End)>();
            stack.Push((0, far));
            stack.Push((far, contour.Length));

            float toleranceSquared = tolerance * tolerance;

            while (stack.Count > 0)
            {
                var (start, end) = stack.Pop();
                var p0 = contour[start];
                var p1 = contour[end % contour.Length];

                int index = -1;
                float maxDistance = toleranceSquared;

                for (int i = start + 1; i < end; i++)
                {
                    float distance = SegmentDistanceSquared(contour[i], p0, p1);

                    if (distance > maxDistance)
                        (index, maxDistance) = (i, distance);
                }

                if (index < 0)
                    continue;

                keep[index] = true;
                stack.Push((start, index));
                stack.Push((index, end));
            }

            var simplified = new List<SKPoint>();

            for (int i = 0; i < contour.Length; i++)
            {
                if (keep[i])
                    simplified.Add(contour[i]);
            }

            return [.. simplified];
        }

        /// <summary>
        /// Marching squares over a grid of samples, padded with <paramref name="outside"/> samples. Sample (i, j) is
        /// mapped to (i * scaleX + offsetX, j * scaleY + offsetY).
        /// </summary>
        private static List<SKPoint[]> TraceCore(ReadOnlySpan<float> field, int width, int height, float iso, float outside,
            float scaleX, float offsetX, float scaleY, float offsetY)
        {
            var contours = new List<SKPoint[]>();

            if (width <= 0 || height <= 0 || field.Length < width * height)
                return contours;

            var grid = new Grid(width, height, iso, outside);

            // Edge ids: horizontal edges between samples (i, j) and (i + 1, j) come first, then vertical edges
            // between (i, j) and (i, j + 1); sample indices start at -1 because the field is padded.
            int horizontalEdges = (width + 1) * (height + 2);
            var state = new TraceState(horizontalEdges + (width + 2) * (height + 1));

            // Inside flags of the two sample rows bordering the current row of cells (padding included).
            var rowAbove = ArrayPool<bool>.Shared.Rent(width + 2);
            var rowBelow = ArrayPool<bool>.Shared.Rent(width + 2);

            try
            {
                Array.Clear(rowBelow, 0, width + 2);

                for (int j = -1; j < height; j++)
                {
                    (rowAbove, rowBelow) = (rowBelow, rowAbove);
                    FillInsideRow(field, grid, j + 1, rowBelow);

                    for (int i = -1; i < width; i++)
                    {
                        bool a = rowAbove[i + 1], b = rowAbove[i + 2], c = rowBelow[i + 2], d = rowBelow[i + 1];

                        // Entirely inside or outside: no outline passes through this cell.
                        if (a == b && b == c && c == d)
                            continue;

                        int cell = (a ? 8 : 0) | (b ? 4 : 0) | (c ? 2 : 0) | (d ? 1 : 0);

                        var edges = new CellEdges(
                            i, j,
                            Top: HorizontalEdge(i, j, width),
                            Bottom: HorizontalEdge(i, j + 1, width),
                            Left: horizontalEdges + VerticalEdge(i, j, width),
                            Right: horizontalEdges + VerticalEdge(i + 1, j, width));

                        switch (cell)
                        {
                            case 1: case 14: AddSegment(state, field, grid, edges, edges.Left, edges.Bottom); break;
                            case 2: case 13: AddSegment(state, field, grid, edges, edges.Bottom, edges.Right); break;
                            case 3: case 12: AddSegment(state, field, grid, edges, edges.Left, edges.Right); break;
                            case 4: case 11: AddSegment(state, field, grid, edges, edges.Top, edges.Right); break;
                            case 6: case 9: AddSegment(state, field, grid, edges, edges.Top, edges.Bottom); break;
                            case 7: case 8: AddSegment(state, field, grid, edges, edges.Top, edges.Left); break;
                            case 5: // top-right and bottom-left inside; the cell centre decides whether they connect
                                if (CentreInside(field, grid, i, j))
                                {
                                    AddSegment(state, field, grid, edges, edges.Top, edges.Left);
                                    AddSegment(state, field, grid, edges, edges.Bottom, edges.Right);
                                }
                                else
                                {
                                    AddSegment(state, field, grid, edges, edges.Top, edges.Right);
                                    AddSegment(state, field, grid, edges, edges.Left, edges.Bottom);
                                }
                                break;
                            case 10: // top-left and bottom-right inside
                                if (CentreInside(field, grid, i, j))
                                {
                                    AddSegment(state, field, grid, edges, edges.Top, edges.Right);
                                    AddSegment(state, field, grid, edges, edges.Left, edges.Bottom);
                                }
                                else
                                {
                                    AddSegment(state, field, grid, edges, edges.Top, edges.Left);
                                    AddSegment(state, field, grid, edges, edges.Bottom, edges.Right);
                                }
                                break;
                        }
                    }
                }

                state.CollectLoops(contours, scaleX, offsetX, scaleY, offsetY);
            }
            finally
            {
                state.Dispose();
                ArrayPool<bool>.Shared.Return(rowAbove);
                ArrayPool<bool>.Shared.Return(rowBelow);
            }

            return contours;
        }

        /// <summary>
        /// Size, level and padding value of the sampled grid.
        /// </summary>
        private readonly record struct Grid(int Width, int Height, float Iso, float Outside);

        /// <summary>
        /// The cell at samples (I, J) .. (I + 1, J + 1) and the ids of its four edges.
        /// </summary>
        private readonly record struct CellEdges(int I, int J, int Top, int Bottom, int Left, int Right);

        /// <summary>
        /// Adds an outline segment between two edges of a cell, creating the crossing points on first use.
        /// </summary>
        private static void AddSegment(TraceState state, ReadOnlySpan<float> field, in Grid grid, in CellEdges cell, int edgeA, int edgeB)
        {
            if (!state.HasPoint(edgeA))
                state.AddPoint(edgeA, CrossingPoint(field, grid, cell, edgeA));

            if (!state.HasPoint(edgeB))
                state.AddPoint(edgeB, CrossingPoint(field, grid, cell, edgeB));

            state.AddSegment(edgeA, edgeB);
        }

        /// <summary>
        /// Where the outline crosses an edge of the cell, in sample coordinates.
        /// </summary>
        private static SKPoint CrossingPoint(ReadOnlySpan<float> field, in Grid grid, in CellEdges cell, int edge)
        {
            int i = cell.I, j = cell.J;

            if (edge == cell.Top)
                return new(i + Crossing(grid, Sample(field, grid, i, j), Sample(field, grid, i + 1, j)), j);

            if (edge == cell.Bottom)
                return new(i + Crossing(grid, Sample(field, grid, i, j + 1), Sample(field, grid, i + 1, j + 1)), j + 1);

            if (edge == cell.Left)
                return new(i, j + Crossing(grid, Sample(field, grid, i, j), Sample(field, grid, i, j + 1)));

            return new(i + 1, j + Crossing(grid, Sample(field, grid, i + 1, j), Sample(field, grid, i + 1, j + 1)));
        }

        /// <summary>
        /// Squared distance from <paramref name="p"/> to the segment <paramref name="a"/>-<paramref name="b"/>.
        /// </summary>
        private static float SegmentDistanceSquared(SKPoint p, SKPoint a, SKPoint b)
        {
            float dx = b.X - a.X, dy = b.Y - a.Y;
            float lengthSquared = dx * dx + dy * dy;
            float t = lengthSquared > 0 ? Math.Clamp(((p.X - a.X) * dx + (p.Y - a.Y) * dy) / lengthSquared, 0f, 1f) : 0f;
            float ex = a.X + t * dx - p.X, ey = a.Y + t * dy - p.Y;

            return ex * ex + ey * ey;
        }

        /// <summary>
        /// Fills the inside flags of sample row <paramref name="y"/> (index 0 is the padding sample at x = -1).
        /// </summary>
        private static void FillInsideRow(ReadOnlySpan<float> field, in Grid grid, int y, bool[] row)
        {
            int width = grid.Width;
            row[0] = false;
            row[width + 1] = false;

            if ((uint)y >= (uint)grid.Height)
            {
                Array.Clear(row, 1, width);
                return;
            }

            var source = field.Slice(y * width, width);

            for (int x = 0; x < width; x++)
                row[x + 1] = source[x] > grid.Iso;
        }

        /// <summary>
        /// Whether the average of the four samples around cell (i, j) is inside; used to resolve saddle cells.
        /// </summary>
        private static bool CentreInside(ReadOnlySpan<float> field, in Grid grid, int i, int j)
            => (Sample(field, grid, i, j) + Sample(field, grid, i + 1, j)
                + Sample(field, grid, i + 1, j + 1) + Sample(field, grid, i, j + 1)) * 0.25f > grid.Iso;

        /// <summary>
        /// Field value at sample (x, y); samples outside the field take the padding value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Sample(ReadOnlySpan<float> field, in Grid grid, int x, int y)
            => (uint)x < (uint)grid.Width && (uint)y < (uint)grid.Height ? field[y * grid.Width + x] : grid.Outside;

        /// <summary>
        /// Fraction of the way from a sample with value <paramref name="from"/> to one with value
        /// <paramref name="to"/> at which the field crosses the iso level.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float Crossing(in Grid grid, float from, float to)
            => Math.Clamp((grid.Iso - from) / (to - from), 0f, 1f);

        // Horizontal edge between samples (i, j) and (i + 1, j); i in [-1, width - 1], j in [-1, height].
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int HorizontalEdge(int i, int j, int width) => (j + 1) * (width + 1) + (i + 1);

        // Vertical edge between samples (i, j) and (i, j + 1); i in [-1, width], j in [-1, height - 1].
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static int VerticalEdge(int i, int j, int width) => (j + 1) * (width + 2) + (i + 1);

        /// <summary>
        /// Crossing points and segments of one trace, indexed by edge id in pooled tables.
        /// </summary>
        private sealed class TraceState : IDisposable
        {
            // Per edge: 1-based index of its crossing point, and the 1-based ids of the two segments meeting there.
            private readonly int[] _edgePoint;
            private readonly int[] _edgeFirst;
            private readonly int[] _edgeSecond;

            private readonly List<SKPoint> _points = [];
            private readonly List<int> _touched = [];
            private readonly List<(int A, int B)> _segments = [];

            public TraceState(int totalEdges)
            {
                _edgePoint = _edgeTablePool.Rent(totalEdges);
                _edgeFirst = _edgeTablePool.Rent(totalEdges);
                _edgeSecond = _edgeTablePool.Rent(totalEdges);
            }

            public bool HasPoint(int edge) => _edgePoint[edge] != 0;

            public void AddPoint(int edge, SKPoint point)
            {
                _points.Add(point);
                _edgePoint[edge] = _points.Count;
                _touched.Add(edge);
            }

            public void AddSegment(int edgeA, int edgeB)
            {
                _segments.Add((edgeA, edgeB));
                Link(edgeA, _segments.Count);
                Link(edgeB, _segments.Count);
            }

            private void Link(int edge, int segment)
            {
                if (_edgeFirst[edge] == 0)
                    _edgeFirst[edge] = segment;
                else
                    _edgeSecond[edge] = segment;
            }

            /// <summary>
            /// Walks the segments into closed loops: every crossing joins exactly two segments.
            /// </summary>
            public void CollectLoops(List<SKPoint[]> contours, float scaleX, float offsetX, float scaleY, float offsetY)
            {
                var used = new bool[_segments.Count + 1];
                var loop = new List<SKPoint>();

                for (int s = 1; s <= _segments.Count; s++)
                {
                    if (used[s])
                        continue;

                    loop.Clear();
                    int startEdge = _segments[s - 1].A;
                    int edge = _segments[s - 1].B;
                    int segment = s;
                    used[s] = true;
                    loop.Add(Point(startEdge));

                    while (edge != startEdge)
                    {
                        loop.Add(Point(edge));

                        int next = _edgeFirst[edge] == segment ? _edgeSecond[edge] : _edgeFirst[edge];

                        if (next == 0 || used[next])
                            break;

                        used[next] = true;
                        segment = next;
                        edge = _segments[next - 1].A == edge ? _segments[next - 1].B : _segments[next - 1].A;
                    }

                    if (loop.Count >= 3)
                        contours.Add([.. loop]);
                }

                SKPoint Point(int edge)
                {
                    var point = _points[_edgePoint[edge] - 1];
                    return new(point.X * scaleX + offsetX, point.Y * scaleY + offsetY);
                }
            }

            /// <summary>
            /// Resets the used entries, so the pooled tables go back zeroed, and returns them.
            /// </summary>
            public void Dispose()
            {
                // Every segment end has a crossing point, so the touched edges cover all used entries.
                foreach (var edge in _touched)
                    _edgePoint[edge] = _edgeFirst[edge] = _edgeSecond[edge] = 0;

                _edgeTablePool.Return(_edgePoint);
                _edgeTablePool.Return(_edgeFirst);
                _edgeTablePool.Return(_edgeSecond);
            }
        }
    }
}
