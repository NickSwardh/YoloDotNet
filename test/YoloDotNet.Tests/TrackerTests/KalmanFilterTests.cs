// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

using YoloDotNet.Trackers;

namespace YoloDotNet.Tests.TrackerTests
{
    public class KalmanFilterTests
    {
        [Fact]
        public void ConstantVelocity_PredictsNextPosition()
        {
            var kalman = new KalmanFilter(0, 0);

            for (int i = 1; i <= 10; i++)
            {
                kalman.Predict();
                kalman.Update(i * 10, 0);
            }

            kalman.Predict();
            var state = kalman.GetState();

            Assert.InRange(state[0], 105f, 115f);
            Assert.InRange(state[1], -2f, 2f);
        }

        [Fact]
        public void StationaryObject_DoesNotDrift()
        {
            var kalman = new KalmanFilter(200, 300);

            for (int i = 0; i < 20; i++)
            {
                kalman.Predict();
                kalman.Update(200, 300);
            }

            kalman.Predict();
            var state = kalman.GetState();

            Assert.InRange(state[0], 199f, 201f);
            Assert.InRange(state[1], 299f, 301f);
        }
    }
}
