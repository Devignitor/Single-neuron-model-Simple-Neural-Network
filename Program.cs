namespace GenerativeAIConsole
{
    class Program
    {
        // Training data (50 items)
        static double[][] inputs =
        {
            new double[] { 1.0, 2.0, 1.0 },
            new double[] { 2.0, 3.0, 2.0 },
            new double[] { 3.0, 4.0, 3.0 },
            new double[] { 4.0, 5.0, 4.0 },
            new double[] { 5.0, 6.0, 5.0 },
            new double[] { 6.0, 7.0, 6.0 },
            new double[] { 7.0, 8.0, 7.0 },
            new double[] { 8.0, 9.0, 8.0 },
            new double[] { 9.0, 10.0, 9.0 },
        };

        static double[] outputs =
        {
            4.0, 7.0, 10.0, 14.0, 16.0, 19.0, 22.0, 25.0, 28.0,
        };

        
        static void Main()
        {
            //Basic Step
            int inputSize = 3;
            double[] weights = new double[inputSize];
            double bias;

            // Initialize weights and bias randomly
            Random rnd = new Random();

            for (int i = 0; i < weights.Length; i++)
                weights[i] = rnd.NextDouble();

            bias = rnd.NextDouble();

            double learningRate = 0.001;
            int epochs = 1000;

            for (int epoch = 0; epoch < epochs; epoch++)
            {
                double totalLoss = 0;

                for (int i = 0; i < inputs.Length; i++)
                {
                    double prediction = Predict(inputs[i]);
                    double error = prediction - outputs[i];

                    // Weight update
                    for (int j = 0; j < weights.Length; j++)
                        weights[j] -= learningRate * error * inputs[i][j];

                    // Bias update
                    bias -= learningRate * error;

                    totalLoss += Loss(prediction, outputs[i]);
                }

                if (epoch % 100 == 0)
                    Console.WriteLine($"Epoch {epoch} | Loss: {totalLoss}");
            }

            double Predict(double[] input)
            {
                double sum = 0;

                for (int i = 0; i < input.Length; i++)
                    sum += input[i] * weights[i];

                sum += bias;

                return sum;
            }

            double Loss(double predicted, double actual)
            {
                double error = predicted - actual;
                return error * error;
            }

            double[] testInput = { 5, 5, 5 };
            double result = Predict(testInput);

            Console.WriteLine($"Prediction for 5 + 5 + 5 = {result}");


            double[] testInput2 = { 20, 5, 6 };
            double result2 = Predict(testInput2);

            Console.WriteLine($"Prediction for 20 + 5 + 6 = {result2}");

            

        }
    }
}