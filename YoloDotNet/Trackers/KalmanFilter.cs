// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Trackers
{
    public class KalmanFilter(float x, float y)
    {
        // [x, y, dx, dy] → position and speed
        private float[] _state = [x, y, 0, 0];

        // Confidence in our guess
        private float[,] _covariance = new float[,]
        {
            { 2, 0, 0, 0 },    // x (more confident)
            { 0, 2, 0, 0 },    // y (more confident)
            { 0, 0, 50, 0 },   // dx (very unsure)
            { 0, 0, 0, 50 }    // dy (very unsure)
        };

        // These control how much we trust motion and measurement
        private readonly float[,] _measurementNoise = new float[,]
        {
            { 0.05f, 0 },
            { 0, 0.05f }
        };

        private readonly float _sigmaP = 0.5f;  // Noise for position
        private readonly float _sigmaV = 0.1f;  // Noise for velocity
        private readonly float _dt = 1f / 30f;  // Time step (1 frame at 30 FPS)

        /// <summary>
        /// Predicts the next state (position and velocity) based on motion.
        /// </summary>
        public void Predict()
        {
            // 1. Use speed to guess next position
            _state[0] += _state[2] * _dt; // x = x + dx * dt
            _state[1] += _state[3] * _dt; // y = y + dy * dt

            // 2. Create a transition matrix F (how state changes over time)
            float[,] f = new float[,]
            {
                { 1, 0, _dt, 0 },
                { 0, 1, 0, _dt },
                { 0, 0, 1, 0 },
                { 0, 0, 0, 1 }
            };

            // 3. Compute new covariance: P = F * P * F^T + Q
            float[,] newCov = Multiply(Multiply(f, _covariance), Transpose(f));
            float[,] q = GetProcessNoise();
            _covariance = Add(newCov, q);
        }

        /// <summary>
        /// Updates the predicted state with a new measurement.
        /// </summary>
        /// <param name="x">Measured X position.</param>
        /// <param name="y">Measured Y position.</param>
        public void Update(float x, float y)
        {
            // Measurement vector z = [x, y]
            float[] z = new float[] { x, y };

            // Measurement matrix H (we only measure x and y)
            float[,] h = new float[,]
            {
                { 1, 0, 0, 0 },
                { 0, 1, 0, 0 }
            };

            // Innovation: y = z - H * x
            float[] yVec = Subtract(z, Multiply(h, _state));

            // S = H * P * H^T + R (measurement prediction)
            float[,] s = Add(Multiply(Multiply(h, _covariance), Transpose(h)), _measurementNoise);

            // Kalman Gain K = P * H^T * S^-1
            float[,] k = Multiply(Multiply(_covariance, Transpose(h)), Inverse2x2(s));

            // New state = old + K * residual
            float[] correction = Multiply(k, yVec);
            _state = Add(_state, correction);

            // New covariance: (I - K * H) * P
            float[,] kh = Multiply(k, h);
            float[,] i = Identity(4);
            float[,] temp = Subtract(i, kh);
            _covariance = Multiply(temp, _covariance);
        }

        /// <summary>
        /// Returns the current estimated state [x, y, dx, dy].
        /// </summary>
        public float[] GetState() => _state;

        /// <summary>
        /// Calculates the process noise matrix (uncertainty from motion).
        /// </summary>
        private float[,] GetProcessNoise()
        {
            float p2 = _sigmaP * _sigmaP;
            float v2 = _sigmaV * _sigmaV;

            return new float[,]
            {
                { p2, 0, _sigmaP * _dt, 0 },
                { 0, p2, 0, _sigmaP * _dt },
                { _sigmaP * _dt, 0, v2, 0 },
                { 0, _sigmaP * _dt, 0, v2 }
            };
        }

        /// <summary>
        /// Multiplies a matrix by a vector.
        /// </summary>
        private static float[] Multiply(float[,] mat, float[] vec)
        {
            float[] result = new float[mat.GetLength(0)];
            for (int i = 0; i < result.Length; i++)
                for (int j = 0; j < vec.Length; j++)
                    result[i] += mat[i, j] * vec[j];
            return result;
        }

        /// <summary>
        /// Multiplies two matrices.
        /// </summary>
        private static float[,] Multiply(float[,] a, float[,] b)
        {
            int rows = a.GetLength(0);
            int cols = b.GetLength(1);
            int shared = a.GetLength(1);
            float[,] result = new float[rows, cols];

            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    for (int k = 0; k < shared; k++)
                        result[i, j] += a[i, k] * b[k, j];

            return result;
        }

        /// <summary>
        /// Adds two matrices together.
        /// </summary>
        private static float[,] Add(float[,] a, float[,] b)
        {
            int rows = a.GetLength(0), cols = a.GetLength(1);
            float[,] result = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, j] = a[i, j] + b[i, j];
            return result;
        }

        /// <summary>
        /// Adds two vectors together.
        /// </summary>
        private static float[] Add(float[] a, float[] b)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
                result[i] = a[i] + b[i];
            return result;
        }

        /// <summary>
        /// Subtracts one vector from another.
        /// </summary>
        private static float[] Subtract(float[] a, float[] b)
        {
            float[] result = new float[a.Length];
            for (int i = 0; i < a.Length; i++)
                result[i] = a[i] - b[i];
            return result;
        }

        /// <summary>
        /// Subtracts one matrix from another.
        /// </summary>
        private static float[,] Subtract(float[,] a, float[,] b)
        {
            int rows = a.GetLength(0), cols = a.GetLength(1);
            float[,] result = new float[rows, cols];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[i, j] = a[i, j] - b[i, j];
            return result;
        }

        /// <summary>
        /// Transposes a matrix (rows become columns).
        /// </summary>
        private static float[,] Transpose(float[,] mat)
        {
            int rows = mat.GetLength(0), cols = mat.GetLength(1);
            float[,] result = new float[cols, rows];
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    result[j, i] = mat[i, j];
            return result;
        }

        /// <summary>
        /// Calculates the inverse of a 2x2 matrix.
        /// </summary>
        private static float[,] Inverse2x2(float[,] mat)
        {
            float a = mat[0, 0], b = mat[0, 1];
            float c = mat[1, 0], d = mat[1, 1];
            float det = a * d - b * c;
            if (Math.Abs(det) < 1e-6f)
                throw new Exception("Matrix is singular and cannot be inverted.");
            float invDet = 1f / det;

            return new float[,]
            {
                { d * invDet, -b * invDet },
                { -c * invDet, a * invDet }
            };
        }

        /// <summary>
        /// Creates an identity matrix of a given size.
        /// </summary>
        private static float[,] Identity(int size)
        {
            float[,] result = new float[size, size];
            for (int i = 0; i < size; i++)
                result[i, i] = 1;
            return result;
        }
    }
}
