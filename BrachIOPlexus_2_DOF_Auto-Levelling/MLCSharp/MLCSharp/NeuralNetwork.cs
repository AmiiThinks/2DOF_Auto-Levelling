using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace MLCSharp
{
    //For testing. Set output type to console application
    public class Test
    {
        public static void Main()
        {
            //Checking with python
            //Output should be -6.16031   ,  4.860726  , -0.67678124
            NeuralNetwork network = new NeuralNetwork(4, 3, new int[] { 4 }, new string[] { "relu", "linear" }, 0, 1);
            MatrixBuilder<double> M = Matrix<double>.Build;

            Console.WriteLine("Starting...");
            Matrix<double> w1 = M.DenseOfArray(new double[,] { { -0.46913165, -0.35420346, 0.12766236, 0.52764505, }, { 0.7143056, 0.7889276, 0.32565337, 0.44582993, }, { 0.44773299, -0.2863869, 0.6054947, 0.15560512, }, { -0.17665333, 0.33406478, 0.6498851, 0.39080018, }, });
            Matrix<double> b1 = M.DenseOfArray(new double[,] { { 0.006911707, 0.0002995945, 0.0, -0.0034662795, } });

            Matrix<double> w2 = M.DenseOfArray(new double[,] { { 0.0005864919, -0.4438249, 0.48540914, }, { 0.25379896, -0.7287866, -0.20937704, }, { -0.72905433, 0.7343912, -0.81658787, }, { -0.814179, 0.86968863, 0.9081696, }, });
            Matrix<double> b2 = M.DenseOfArray(new double[,] { { 0.0, -0.00010902006, 0.014027984, } });


            network.setWeights(new Matrix<double>[] { w1, w2 }, new Matrix<double>[] { b1, b2 });

            Matrix<double> input = Matrix<double>.Build.Dense(1, 4);
            input[0, 0] = 1;
            input[0, 1] = 2;
            input[0, 2] = 3;
            input[0, 3] = 4;
            Console.WriteLine(network.predict(input).ToString());


            //NeuralNetwork network = new NeuralNetwork(2, 1, new int[] { 4 }, new string[] { "sigmoid", "sigmoid" }, 0.5, 1);
            ////network.printWeights();
            //MatrixBuilder<double> M = Matrix<double>.Build;
            //double[][,] X_array = new double[4][,];
            //X_array[0] = new double[,] { { 0, 0 } };
            //X_array[1] = new double[,] { { 0, 1 } };
            //X_array[2] = new double[,] { { 1, 0 } };
            //X_array[3] = new double[,] { { 1, 1 } };

            //double[][,] y_array = new double[4][,];
            //y_array[0] = new double[,] { { 0 } };
            //y_array[1] = new double[,] { { 1 } };
            //y_array[2] = new double[,] { { 1 } };
            //y_array[3] = new double[,] { { 0 } };

            //int epochs = 10000;

            //for (int i = 0; i < epochs; i++)
            //{
            //    int index = i % 4;
            //    double error = network.train(M.DenseOfArray(X_array[index]), M.DenseOfArray(y_array[index]));
            //    //Console.WriteLine(error);
            //    //network.printWeights();

            //}
            //Console.WriteLine("Predict 0 1");
            //Console.WriteLine(network.predict(M.DenseOfArray(new double[,] { { 0, 1 } }))[0, 0]);

            //Console.WriteLine("Check copy function");
            //NeuralNetwork newNetwork = new NeuralNetwork(network);
            //double o1 = network.predict(M.DenseOfArray(new double[,] { { 0, 1 } }))[0, 0];
            //double o2 = newNetwork.predict(M.DenseOfArray(new double[,] { { 0, 1 } }))[0, 0];
            //Console.WriteLine(o1);
            //Console.WriteLine(o2);
            //Console.WriteLine(o1 == o2 ? "Pass" : "Fail");

            //network.train(M.DenseOfArray(X_array[0]), M.DenseOfArray(y_array[0]));

            //o1 = network.predict(M.DenseOfArray(new double[,] { { 0, 1 } }))[0, 0];
            //o2 = newNetwork.predict(M.DenseOfArray(new double[,] { { 0, 1 } }))[0, 0];
            //Console.WriteLine(o1);
            //Console.WriteLine(o2);
            //Console.WriteLine(o1 != o2 ? "Pass" : "Fail");


            Console.ReadKey();
        }
    }

    public class NeuralNetwork
    {
        public int[] layers { get; private set; }
        public Matrix<double>[] weights { get; private set; }
        public Matrix<double>[] biases { get; private set; }
        public double[][] hiddenVals { get; private set; } //z = wx + b
        public double[][] activationVals { get; private set; }//a = relu(z)
        public string[] activationFunctions { get; private set; }
        public double learningRate { get; private set; }
        public int intermediate_layer { get; private set; }

        public Matrix<double> intermediate_value { get; private set; }

        private MatrixBuilder<double> M = Matrix<double>.Build;


        public NeuralNetwork(int inputSize, int outputSize, int[] hiddenLayers, string[] activationFunctions, double learningRate, int intermediate_layer)
        {
            this.activationFunctions = activationFunctions; //list of activation functions used by each layer (including output layer). If you don't want activation on the output layer, specify 'linear'
            this.learningRate = learningRate;
            this.intermediate_layer = intermediate_layer;
            this.intermediate_value = M.DenseOfArray(new double[,] { { 0, 0, 0, 0 } });

            layers = new int[2 + hiddenLayers.Length];
            layers[0] = inputSize;
            for (int i = 0; i < hiddenLayers.Length; i++)
            {
                layers[i + 1] = hiddenLayers[i];
            }
            layers[layers.Length - 1] = outputSize;

            weights = new Matrix<double>[layers.Length - 1];
            biases = new Matrix<double>[layers.Length - 1];
            hiddenVals = new double[layers.Length - 1][];
            activationVals = new double[layers.Length - 1][];

            for (int i = 0; i < weights.Length; i++)
            {
                //Initialize weights and biases
                Matrix<double> normalDistr = M.Random(layers[i], layers[i + 1]);
                weights[i] = M.Dense(layers[i], layers[i + 1]);//initialize weight matrix so it is the right size
                normalDistr.Map(x => Math.Min(Math.Max(x, -1), 1), weights[i]);//truncate normal distribution

                biases[i] = M.Dense(1, layers[i + 1], 1.0);

                //Initialize arrays to store z and activation values for backpropagation
                hiddenVals[i] = new double[layers[i + 1]];
                activationVals[i] = new double[layers[i + 1]];
            }

        }
        //Copy function
        public NeuralNetwork(NeuralNetwork network)
        {
            layers = new int[network.layers.Length];
            network.layers.CopyTo(layers, 0);

            activationFunctions = new string[network.activationFunctions.Length];
            network.activationFunctions.CopyTo(activationFunctions, 0);

            learningRate = network.learningRate;

            weights = new Matrix<double>[layers.Length - 1];
            biases = new Matrix<double>[layers.Length - 1];
            hiddenVals = new double[layers.Length - 1][];
            activationVals = new double[layers.Length - 1][];

            for (int i = 0; i < weights.Length; i++)
            {
                //Initialize weights and biases
                weights[i] = M.Dense(layers[i], layers[i + 1]);//initialize weight matrix so it is the right size
                network.weights[i].CopyTo(weights[i]);//Copy weights from input network

                biases[i] = M.Dense(1, layers[i + 1], 1.0);//initialize bias matrix so it is the right size
                network.biases[i].CopyTo(biases[i]);//Copy biases from input network

                //Initialize arrays to store z and activation values for backpropagation
                hiddenVals[i] = new double[layers[i + 1]];
                activationVals[i] = new double[layers[i + 1]];
            }
        }

        public void copyWeights(NeuralNetwork sourceNetwork)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                sourceNetwork.weights[i].CopyTo(weights[i]);//Copy weights from input network
                sourceNetwork.biases[i].CopyTo(biases[i]);//Copy biases from input network
            }
        }

        public void setWeights(Matrix<double>[] weights, Matrix<double>[] biases)
        {
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i].CopyTo(this.weights[i]);//Set weights
                biases[i].CopyTo(this.biases[i]);//Set biases
            }
        }

        public Matrix<double> predict(Matrix<double> input)
        {
            Matrix<double> a = input; //First layer, multiply input by weight
            Matrix<double> z;
            for (int i = 0; i < weights.Length; i++)
            {
                if (i == intermediate_layer)
                {
                    intermediate_value = a;
                }
                //z = wX + b
                z = a.Multiply(weights[i]);
                z = z.Add(biases[i]);

                a = M.Dense(z.RowCount, z.ColumnCount); //Resize a so we can output activation to it

                //Compute activation
                for (int j = 0; j < z.ColumnCount; j++)
                {
                    //compute activaiton based on what layer we're in
                    a[0, j] = activation(z[0, j], i);

                    //Record for backprop
                    hiddenVals[i][j] = z[0, j];
                    activationVals[i][j] = a[0, j];
                }
            }
            return a;
        }

        public void backwardProp(Matrix<double> input, Matrix<double> pred, Matrix<double> label)
        {
            int L = layers.Length - 2; //for indexing weights and activations
            double[][] grad = new double[layers.Length - 1][]; //for storing gradient.

            for (int i = 0; i < grad.Length; i++)
            {
                grad[i] = new double[weights[i].ColumnCount]; //for each layer, gradient size will match the number of neurons
            }

            for (int j = 0; j < weights[L].ColumnCount; j++)
            {
                double errorPartial = (activationVals[L][j] - label[0, j]);
                errorPartial = Math.Max(Math.Min(errorPartial, 10), -10);
                grad[L][j] = activationPrime(hiddenVals[L][j], L) * errorPartial; //For last layer, store derivative of activation * gradient of error
                biases[L][0, j] -= grad[L][j] * learningRate; //update bias
                for (int i = 0; i < weights[L].RowCount; i++)
                {
                    weights[L][i, j] -= grad[L][j] * activationVals[L - 1][i] * learningRate; //update weights
                }
            }

            L--; //Move one layer back

            for (; L >= 0; L--)//For the rest of the layers
            {
                for (int j = 0; j < weights[L].ColumnCount; j++)
                {
                    //Compute grad (sum of grad times weights from next layer all multiplied by derivative of activaiton)
                    grad[L][j] = 0;
                    for (int k = 0; k < weights[L+1].ColumnCount; k++)
                    {
                        grad[L][j] += weights[L + 1][j, k] * grad[L + 1][k];
                    }
                    grad[L][j] *= activationPrime(hiddenVals[L][j], L);

                    biases[L][0, j] -= grad[L][j] * learningRate;//update bias
                    for (int i = 0; i < weights[L].RowCount; i++)
                    {
                        if (L == 0)//If first layer (layer after input) then "activation" is actually the input features
                        {
                            weights[L][i, j] -= grad[L][j] * input[0, i] * learningRate;//update weights
                        }
                        else
                        {
                            weights[L][i, j] -= grad[L][j] * activationVals[L - 1][i] * learningRate;//update weights
                        }
                    }
                } 
            }
        }

        //Backward prop for actor network
        public void backwardProp(Matrix<double> input, double tdLoss, double mu, double sigma, double sample)
        {
            int L = layers.Length - 2; //for indexing weights and activations
            double[][] grad = new double[layers.Length - 1][]; //for storing gradient.

            for (int i = 0; i < grad.Length; i++)
            {
                grad[i] = new double[weights[i].ColumnCount]; //for each layer, gradient size will match the number of neurons
            }

            for (int j = 0; j < weights[L].ColumnCount; j++)
            {
                //Calculate derivative of error with respect to last activation (nn output)
                double dEda;
                switch (j)
                {
                    case 0:
                        dEda = -tdLoss / (mu + sigma * sample);
                        break;
                    case 1:
                        dEda = -tdLoss / (mu + sigma * sample) * sample;
                        break;
                    default:
                        dEda = 0;
                        break;
                }
                grad[L][j] = activationPrime(hiddenVals[L][j], L) * dEda; //For last layer, store derivative of activation * gradient of error
                biases[L][0, j] -= grad[L][j] * learningRate; //update bias
                for (int i = 0; i < weights[L].RowCount; i++)
                {
                    weights[L][i, j] -= grad[L][j] * activationVals[L - 1][i] * learningRate; //update weights
                }
            }

            L--; //Move one layer back

            for (; L >= 0; L--)//For the rest of the layers
            {
                for (int j = 0; j < weights[L].ColumnCount; j++)
                {
                    //Compute grad (sum of grad times weights from next layer all multiplied by derivative of activaiton)
                    grad[L][j] = 0;
                    for (int k = 0; k < weights[L + 1].ColumnCount; k++)
                    {
                        grad[L][j] += weights[L + 1][j, k] * grad[L + 1][k];
                    }
                    grad[L][j] *= activationPrime(hiddenVals[L][j], L);

                    biases[L][0, j] -= grad[L][j] * learningRate;//update bias
                    for (int i = 0; i < weights[L].RowCount; i++)
                    {
                        if (L == 0)//If first layer (layer after input) then "activation" is actually the input features
                        {
                            weights[L][i, j] -= grad[L][j] * input[0, i] * learningRate;//update weights
                        }
                        else
                        {
                            weights[L][i, j] -= grad[L][j] * activationVals[L - 1][i] * learningRate;//update weights
                        }
                    }
                }
            }
        }

        public double train(Matrix<double> X, Matrix<double> y)
        {
            Matrix<double> pred = predict(X);
            double error = 0;
            for (int i = 0; i < y.ColumnCount; i++)
            {
                error += 0.5 * Math.Pow((y[0, i] - pred[0, i]), 2);
            }
            backwardProp(X, pred, y);
            return error;
        }

        public double train(Matrix<double> X, Matrix<double> pred, Matrix<double> y)
        {
            //Train without predicting in case it was already predicted before
            double error = 0;
            for (int i = 0; i < y.ColumnCount; i++)
            {
                error += 0.5 * Math.Pow((y[0, i] - pred[0, i]), 2);
            }
            backwardProp(X, pred, y);
            return error;
        }

        //Training for actor
        public double trainActor(Matrix<double> X, double tdLoss, double mu, double sigma, double sample)
        {
            double error = -Math.Log(mu + sigma * sample) * tdLoss;
            backwardProp(X, tdLoss, mu, sigma, sample);
            return error;
            
        }

        public double activation(double z, int layer)
        {
            switch (activationFunctions[layer])
            {
                case "relu":
                    return relu(z);
                case "leakyRelu":
                    return leakyRelu(z);
                case "sigmoid":
                    return sigmoid(z);
                case "linear":
                    return z;
                default:
                    return z;
            }
        }

        public double activationPrime(double z, int layer)
        {
            switch (activationFunctions[layer])
            {
                case "relu":
                    return reluPrime(z);
                case "leakyRelu":
                    return leakyReluPrime(z);
                case "sigmoid":
                    return sigmoidPrime(z);
                case "linear":
                    return 1;
                default:
                    return 1;
            }
        }

        public double sigmoid(double z)
        {
            return (1 / (1 + Math.Exp(-z)));
        }

        public double sigmoidPrime(double z)
        {
            return sigmoid(z) * (1 - sigmoid(z));
        }
        public double leakyRelu(double z)
        {
            if (z > 0)
            {
                return z;
            }
            else
            {
                return z * 0.3;
            }
        }
        public double leakyReluPrime(double output)
        {
            if (output > 0)
            {
                return 1;
            }
            else
            {
                return 0.3;
            }
        }

        public double relu(double z)
        {
            if (z > 0)
            {
                return z;
            }
            else
            {
                return 0;
            }
        }

        public double reluPrime(double output)
        {
            if (output > 0)
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }

        public void printWeights()
        {
            Console.WriteLine("Weights");
            foreach (Matrix<double> w in weights)
            {
                Console.WriteLine(w.ToString());
            }
        }
        public void printBiases()
        {
            Console.WriteLine("Biases");
            foreach (Matrix<double> b in biases)
            {
                Console.WriteLine(b.ToString());
            }
        }
    }
}
