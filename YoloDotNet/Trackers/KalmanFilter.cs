// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2025-2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Trackers
{
    /// <summary>
    /// Constant-velocity Kalman filter that tracks a 2D point (e.g. a bounding box center).
    /// Position is in pixels and velocity in pixels per frame.
    /// </summary>
    /// <remarks>
    /// The X and Y axes don't influence each other in this model, so the filter runs as two
    /// independent 1D filters. That gives the same result as the full 4x4 matrix form,
    /// without allocating matrices or inverting them every frame.
    /// </remarks>
    public class KalmanFilter(float x, float y)
    {
        // Initial uncertainty (variance): the position comes from a detection, the velocity is unknown.
        private const float InitialPositionVariance = 10f;
        private const float InitialVelocityVariance = 1000f;

        // How much the detector's box center jitters between frames (px²).
        private const float MeasurementNoise = 4f;

        // How much the velocity may change between frames (acceleration noise).
        private const float ProcessNoise = 1f;

        private AxisFilter _x = new(x);
        private AxisFilter _y = new(y);

        private readonly float[] _state = new float[4];

        /// <summary>
        /// Moves the estimate one frame ahead using the current velocity.
        /// Call once per frame, before <see cref="Update"/>.
        /// </summary>
        public void Predict()
        {
            _x.Predict();
            _y.Predict();
        }

        /// <summary>
        /// Corrects the estimate with a measured position.
        /// </summary>
        /// <param name="x">Measured X position.</param>
        /// <param name="y">Measured Y position.</param>
        public void Update(float x, float y)
        {
            _x.Update(x);
            _y.Update(y);
        }

        /// <summary>
        /// Returns the current estimate as [x, y, dx, dy].
        /// The returned array is reused by the filter; copy it if you need to keep the values.
        /// </summary>
        public float[] GetState()
        {
            _state[0] = _x.Position;
            _state[1] = _y.Position;
            _state[2] = _x.Velocity;
            _state[3] = _y.Velocity;
            return _state;
        }

        /// <summary>
        /// 1D constant-velocity Kalman filter (time step = 1 frame).
        /// Covariance is the symmetric 2x2 matrix [[pp, pv], [pv, vv]].
        /// </summary>
        private struct AxisFilter(float position)
        {
            public float Position = position;
            public float Velocity = 0f;

            private float _pp = InitialPositionVariance;
            private float _pv = 0f;
            private float _vv = InitialVelocityVariance;

            public void Predict()
            {
                Position += Velocity;

                // P = F·P·Fᵀ + Q, with F = [[1, 1], [0, 1]] and Q from a random-acceleration model
                _pp += 2f * _pv + _vv + ProcessNoise / 4f;
                _pv += _vv + ProcessNoise / 2f;
                _vv += ProcessNoise;
            }

            public void Update(float measurement)
            {
                // Innovation variance is always > 0 since MeasurementNoise > 0, so no division by zero
                var innovationVariance = _pp + MeasurementNoise;
                var positionGain = _pp / innovationVariance;
                var velocityGain = _pv / innovationVariance;

                var residual = measurement - Position;
                Position += positionGain * residual;
                Velocity += velocityGain * residual;

                // P = (I - K·H)·P
                _vv -= velocityGain * _pv;
                _pv *= 1f - positionGain;
                _pp *= 1f - positionGain;
            }
        }
    }
}
