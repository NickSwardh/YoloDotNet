namespace YoloDotNet.Trackers
{
    public class KalmanFilter
    {
        private float[] state;        // State vector [x, y, dx, dy]
        private float[,] covariance;  // Covariance matrix
        private readonly float[,] processNoise;      // Process noise
        private readonly float[,] measurementNoise;  // Measurement noise

        private float sigmaP = 0.5f; // Process noise for position
        private float sigmaV = 0.1f; // Process noise for velocity

        public KalmanFilter(float x, float y)
        {
            // Initial state [x, y, vx, vy]
            state = new float[] { x, y, 0, 0 };

            // Matrix for storing the uncertanty.
            // High values = High uncertainty → We are not sure if our position and velocity are correct.
            // Low values = Low uncertainty → We are very sure that our position and velocity are correct.
            covariance = new float[,]
            {
                { 1, 0, 0, 0 }, // x coordinate
                { 0, 1, 0, 0 }, // y coordinate
                { 0, 0, 1, 0 }, // x-velocity
                { 0, 0, 0, 1 }  // y-velocity
            };

            // The Process Noise matrix compensates for real - world movement errors.
            // Larger values = more flexibility(good for erratic motion).
            // Smaller values = stricter tracking(good for smooth motion).
            float dt = 1f / 30f; // Time step (frame interval)

            // Process noise for position (higher means more trust in new measurements)
            float sigmaP = 0.5f;

            // Process noise for velocity (lower means smoother tracking)
            float sigmaV = 0.1f;

            // Compute squared values
            float sigmaP2 = sigmaP * sigmaP;
            float sigmaV2 = sigmaV * sigmaV;

            processNoise = new float[,]
            {
                { sigmaP2, 0, sigmaP * dt, 0 },
                { 0, sigmaP2, 0, sigmaP * dt },
                { sigmaP * dt, 0, sigmaV2, 0 },
                { 0, sigmaP * dt, 0, sigmaV2 }
            };

            measurementNoise = new float[,]
            {
                { 0.05f, 0 },
                { 0, 0.05f }
            };
        }

        /// <summary>
        /// Predicts the next position.
        /// </summary>
        public void Predict()
        {
            /*
            // Apply transition matrix: state = F * state
            float newX = state[0] + state[2]; // x + vx
            float newY = state[1] + state[3]; // y + vy
            state[0] = newX;
            state[1] = newY;

            // Increase uncertainty in covariance
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    covariance[i, j] += processNoise[i, j];
            */
            
            // Apply transition matrix: state = F * state
            float newX = state[0] + state[2]; // x + vx
            float newY = state[1] + state[3]; // y + vy
            state[0] = newX;
            state[1] = newY;
            
            // Adjust process noise based on speed of object
            float speed = (float)Math.Sqrt(state[2] * state[2] + state[3] * state[3]);
            float dynamicSigmaP = sigmaP * (1 + speed / 10);  // Adjust process noise based on speed
            float dynamicSigmaV = sigmaV * (1 - speed / 10);  // Lower noise for fast objects

            // Dynamically adjust process noise for fast-moving objects
            processNoise[0, 0] = dynamicSigmaP * dynamicSigmaP;
            processNoise[1, 1] = dynamicSigmaP * dynamicSigmaP;
            processNoise[2, 2] = dynamicSigmaV * dynamicSigmaV;
            processNoise[3, 3] = dynamicSigmaV * dynamicSigmaV;
            processNoise[0, 2] = dynamicSigmaP * 1f / 30f;
            processNoise[1, 3] = dynamicSigmaP * 1f / 30f;
            processNoise[2, 0] = dynamicSigmaP * 1f / 30f;
            processNoise[3, 1] = dynamicSigmaP * 1f / 30f;

            // Increase uncertainty in covariance
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    covariance[i, j] += processNoise[i, j];
        }

        /// <summary>
        /// Updates the state with a new measurement.
        /// </summary>
        public void Update(float x, float y)
        {
            // Compute Kalman gain K = P * H' * (H * P * H' + R)^-1
            float S00 = covariance[0, 0] + measurementNoise[0, 0]; // Innovation covariance
            float S11 = covariance[1, 1] + measurementNoise[1, 1];
            float K00 = covariance[0, 0] / S00;
            float K10 = covariance[1, 1] / S11;
            float K20 = covariance[2, 0] / S00;
            float K30 = covariance[3, 1] / S11;

            // Update state: new state = old state + K * residual
            float residualX = x - state[0];
            float residualY = y - state[1];

            state[0] += K00 * residualX;
            state[1] += K10 * residualY;
            state[2] += K20 * residualX;
            state[3] += K30 * residualY;

            // Update covariance: P = (I - K * H) * P
            covariance[0, 0] *= (1 - K00);
            covariance[1, 1] *= (1 - K10);
        }

        public float[] GetState() => state;
    }
}
