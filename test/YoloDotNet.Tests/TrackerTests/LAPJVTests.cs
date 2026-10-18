// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using YoloDotNet.Exceptions;
using YoloDotNet.Trackers;

namespace YoloDotNet.Tests.TrackerTests
{
    public class LAPJVTests
    {
        [Fact]
        public void MoreRowsThanColumns_AssignsCheapestRow()
        {
            var assignment = LAPJV.Solve(new float[,] { { 1.7f }, { 0.46f } });

            Assert.Equal([-1, 0], assignment);
        }

        [Fact]
        public void MoreColumnsThanRows_AssignsCheapestColumn()
        {
            var assignment = LAPJV.Solve(new float[,] { { 0.9f, 0.1f, 0.5f } });

            Assert.Equal([1], assignment);
        }

        [Fact]
        public void NonFiniteCost_Throws()
        {
            Assert.Throws<YoloDotNetException>(() => LAPJV.Solve(new float[,] { { float.NaN, 0.1f } }));
        }

        [Fact]
        public void InputMatrix_IsNotModified()
        {
            var matrix = new float[,] { { 0.3f, 0.9f }, { 0.4f, 0.2f } };
            var copy = (float[,])matrix.Clone();

            LAPJV.Solve(matrix);

            Assert.Equal(copy, matrix);
        }

        [Fact]
        public void RandomMatrices_MatchBruteForceOptimum()
        {
            var rnd = new Random(1);

            for (int t = 0; t < 2000; t++)
            {
                int rows = rnd.Next(1, 7), cols = rnd.Next(1, 7);
                var matrix = new float[rows, cols];

                for (int i = 0; i < rows; i++)
                    for (int j = 0; j < cols; j++)
                        matrix[i, j] = (float)rnd.NextDouble();

                var assignment = LAPJV.Solve((float[,])matrix.Clone());

                Assert.Equal(Math.Min(rows, cols), assignment.Count(a => a >= 0));
                Assert.Equal(assignment.Count(a => a >= 0), assignment.Where(a => a >= 0).Distinct().Count());

                float total = 0;
                for (int i = 0; i < rows; i++)
                    if (assignment[i] >= 0)
                        total += matrix[i, assignment[i]];

                Assert.Equal(BruteForce(matrix, rows, cols), total, 4);
            }
        }

        private static float BruteForce(float[,] m, int rows, int cols)
        {
            // Minimum total cost with min(rows, cols) pairs
            float best = float.MaxValue;
            int pairs = Math.Min(rows, cols);

            void Search(int i, int assigned, bool[] usedCols, float sum)
            {
                if (assigned == pairs)
                {
                    best = Math.Min(best, sum);
                    return;
                }

                if (i == rows || rows - i < pairs - assigned)
                    return;

                if (rows - i > pairs - assigned)
                    Search(i + 1, assigned, usedCols, sum);

                for (int j = 0; j < cols; j++)
                {
                    if (usedCols[j])
                        continue;

                    usedCols[j] = true;
                    Search(i + 1, assigned + 1, usedCols, sum + m[i, j]);
                    usedCols[j] = false;
                }
            }

            Search(0, 0, new bool[cols], 0);
            return best;
        }
    }
}
