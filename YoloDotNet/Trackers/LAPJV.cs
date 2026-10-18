// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Trackers
{
    /// <summary>
    /// Linear assignment solver for rectangular cost matrices (shortest augmenting path, Jonker-Volgenant / Hungarian style).
    /// Finds the best way to match items from two groups with the lowest total cost.
    /// The "cost" shows how bad a match is; a lower cost means a better match.
    /// </summary>
    public static class LAPJV
    {
        /// <summary>
        /// Solves the linear assignment problem minimizing total cost.
        /// </summary>
        /// <param name="costMatrix">Input cost matrix (rows x columns). Not modified.</param>
        /// <returns>
        /// Assignment array: assigned column index for each row (or -1 if unassigned).
        /// The returned array length equals the number of input rows.
        /// Every row is assigned when rows &lt;= columns; otherwise the extra rows are left unassigned.
        /// </returns>
        /// <exception cref="YoloDotNetException">Thrown when the matrix is null, empty or contains NaN/infinite values.</exception>
        public static int[] Solve(float[,] costMatrix)
        {
            if (costMatrix == null)
                throw new YoloDotNetException(nameof(costMatrix));

            int nRows = costMatrix.GetLength(0);
            int nCols = costMatrix.GetLength(1);

            if (nRows == 0 || nCols == 0)
                throw new YoloDotNetException("Cost matrix must have non-zero dimensions.");

            // NaN or infinite costs would prevent the search below from ever finding a path
            foreach (var value in costMatrix)
            {
                if (float.IsFinite(value) is false)
                    throw new YoloDotNetException("Cost matrix must only contain finite values.");
            }

            // The algorithm requires rows <= columns; transpose otherwise
            bool transposed = nRows > nCols;
            int n = transposed ? nCols : nRows;
            int m = transposed ? nRows : nCols;

            double Cost(int i, int j) => transposed ? costMatrix[j, i] : costMatrix[i, j];

            // Potentials (u for rows, v for columns) keep track of how much cheaper each row/column
            // can be made, so the search only has to follow zero "reduced cost" edges.
            // Arrays are 1-based; index 0 is a virtual column used as the start of each search.
            var u = new double[n + 1];
            var v = new double[m + 1];
            var p = new int[m + 1];         // p[j] = row assigned to column j (0 = none)
            var way = new int[m + 1];       // Previous column on the current search path
            var minv = new double[m + 1];   // Smallest reduced cost found so far per column
            var used = new bool[m + 1];     // Columns already visited in the current search

            for (int i = 1; i <= n; i++)
            {
                // Search for the cheapest path from row i to a free column (Dijkstra-like)
                p[0] = i;
                int j0 = 0;

                Array.Fill(minv, double.PositiveInfinity);
                Array.Fill(used, false);

                do
                {
                    used[j0] = true;
                    int i0 = p[j0];
                    double delta = double.PositiveInfinity;
                    int j1 = 0;

                    for (int j = 1; j <= m; j++)
                    {
                        if (used[j])
                            continue;

                        double reduced = Cost(i0 - 1, j - 1) - u[i0] - v[j];

                        if (reduced < minv[j])
                        {
                            minv[j] = reduced;
                            way[j] = j0;
                        }

                        if (minv[j] < delta)
                        {
                            delta = minv[j];
                            j1 = j;
                        }
                    }

                    // Can't happen with finite costs and rows <= columns, but never loop forever
                    if (j1 == 0)
                        throw new YoloDotNetException("Failed to solve the assignment problem.");

                    for (int j = 0; j <= m; j++)
                    {
                        if (used[j])
                        {
                            u[p[j]] += delta;
                            v[j] -= delta;
                        }
                        else
                        {
                            minv[j] -= delta;
                        }
                    }

                    j0 = j1;
                }
                while (p[j0] != 0);

                // Walk the path back and shift assignments along it, which assigns row i
                do
                {
                    int j1 = way[j0];
                    p[j0] = p[j1];
                    j0 = j1;
                }
                while (j0 != 0);
            }

            var assignment = new int[nRows];
            Array.Fill(assignment, -1);

            for (int j = 1; j <= m; j++)
            {
                if (p[j] == 0)
                    continue;

                int row = p[j] - 1;
                int col = j - 1;

                if (transposed)
                    assignment[col] = row;
                else
                    assignment[row] = col;
            }

            return assignment;
        }
    }
}
