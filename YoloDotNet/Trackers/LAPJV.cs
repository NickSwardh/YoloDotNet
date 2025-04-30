namespace YoloDotNet.Trackers
{
    using System;

    public static class LAPJV
    {
        public static int[] Solve(float[,] costMatrix)
        {
            int n = costMatrix.GetLength(0);
            int m = costMatrix.GetLength(1);
            float[] u = new float[n];
            float[] v = new float[m];
            int[] rowsol = new int[n];
            int[] colsol = new int[m];
            bool[] committed = new bool[n];

            // Initialize the solution arrays to -1
            Array.Fill(rowsol, -1);
            Array.Fill(colsol, -1);

            // Step 1: Subtract row minimum
            for (int i = 0; i < n; i++)
            {
                float minVal = float.MaxValue;
                for (int j = 0; j < m; j++)
                    if (costMatrix[i, j] < minVal)
                        minVal = costMatrix[i, j];

                u[i] = minVal;
                for (int j = 0; j < m; j++)
                    costMatrix[i, j] -= minVal;
            }

            // Step 2: Subtract column minimum
            for (int j = 0; j < m; j++)
            {
                float minVal = float.MaxValue;
                for (int i = 0; i < n; i++)
                    if (costMatrix[i, j] < minVal)
                        minVal = costMatrix[i, j];

                v[j] = minVal;
                for (int i = 0; i < n; i++)
                    costMatrix[i, j] -= minVal;
            }

            // Step 3: Find initial assignments
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < m; j++)
                {
                    if (costMatrix[i, j] == 0 && colsol[j] == -1)
                    {
                        rowsol[i] = j;
                        colsol[j] = i;
                        break;
                    }
                }
            }

            // Step 4: Find free rows
            for (int i = 0; i < n; i++)
                committed[i] = rowsol[i] == -1;

            // Step 5: Augmenting path algorithm
            int free = 0;
            while (free < n)
            {
                int minRow = -1;
                float minVal = float.MaxValue;
                for (int i = 0; i < n; i++)
                {
                    if (committed[i])
                    {
                        for (int j = 0; j < m; j++)
                        {
                            if (costMatrix[i, j] < minVal)
                            {
                                minVal = costMatrix[i, j];
                                minRow = i;
                            }
                        }
                    }
                }

                if (minRow == -1)
                    break;

                committed[minRow] = false;
                int minCol = -1;
                for (int j = 0; j < m; j++)
                {
                    if (costMatrix[minRow, j] == 0 && colsol[j] == -1)
                    {
                        rowsol[minRow] = j;
                        colsol[j] = minRow;
                        free++;
                        break;
                    }
                }
            }

            return rowsol;
        }
    }
}