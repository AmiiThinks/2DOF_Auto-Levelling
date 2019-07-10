using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;

namespace MLCSharp
{
    
    public class RLAgent
    {
        public double gamma { get; private set; } //discount factor
        public NeuralNetwork actor { get; private set; }
        public NeuralNetwork critic { get; private set; }
        public Random rnd { get; set; }

        private Matrix<double> prevState;

        private double mu;
        private double sigma;
        private double sample;

        //Actor critic
        //https://medium.com/@asteinbach/actor-critic-using-deep-rl-continuous-mountain-car-in-tensorflow-4c1fb2110f7c
        public RLAgent(int inputSizeCritic, int outputSizeCritic, int[] hiddenLayersCritic, string[] activationsCritic, double lrCritic, 
            int intputSizeActor, int outputSizeActor, int[] hiddenLayersActor, string[] activationsActor, double lrActor, double gamma)
        {
            this.gamma = gamma;
            rnd = new Random();
            actor = new NeuralNetwork(intputSizeActor, outputSizeActor, hiddenLayersActor, activationsActor, lrActor);
            critic = new NeuralNetwork(inputSizeCritic, outputSizeCritic, hiddenLayersCritic, activationsCritic, lrCritic);
        }

        public Matrix<double> arrayToMatrix(double[] input)
        {
            Matrix<double> output = Matrix<double>.Build.Dense(1, input.Length);
            for (int i = 0; i < input.Length; i++)
            {
                output[0, i] = input[i];
            }
            return output;
        }
        public double SelectAction(double[] state)
        {
            Matrix<double> stateM = arrayToMatrix(state);
            Vector<double> actorOutput = actor.predict(stateM).Row(0);
            prevState = arrayToMatrix(state);
            //Reparameterization trick for Gaussian Distribution
            //https://math.stackexchange.com/questions/2540170/reparameterization-trick-for-gaussian-distribution/2555256

            mu = actorOutput[0]; //stdev
            sigma = actorOutput[1]; //mean
            sample = Normal.Sample(0, 1);
            Console.WriteLine("Mu: " + mu + " Sigma: " + sigma + " Sample: " + sample);

            return mu + sigma * sample;
        }

        public void Update(double[] newState, double reward)
        { 
            //Get value of new state
            double newValue = reward + gamma * critic.predict(arrayToMatrix(newState))[0, 0];

            //Get value of prev state
            Matrix<double> prevValue = critic.predict(prevState);

            //Train critic on difference between prev state value and new state value
            double error = critic.train(prevState, prevValue, arrayToMatrix(new double[] { newValue }));
            Console.WriteLine("Critic erorr: " + error);

            //Calculate TD loss
            double tdLoss = 0.5 * Math.Pow((newValue - prevValue[0, 0]), 2);
            //Train actor
            double errorActor = actor.trainActor(prevState, tdLoss, mu, sigma, sample);
            Console.WriteLine("Actor error: " + errorActor);


        }

        //public double gamma { get; private set; } //discount factor
        //public double epsilon { get; private set; } //exploration rate
        //public int c { get; private set; } //steps before updating target network with predictio network
        //public int current_steps { get; private set; } //current number of steps
        //public NeuralNetwork predNetwork { get; private set; }
        //public NeuralNetwork targetNetwork { get; private set; }
        //public Random rnd { get; set; }

        //public RLAgent(double gamma, double epsilon, int c, double nnLearningRate)
        //{
        //    this.gamma = gamma;
        //    this.epsilon = epsilon;
        //    this.c = c;
        //    current_steps = 0;
        //    rnd = new Random();
        //    predNetwork = new NeuralNetwork(6, 9, new int[] { 10 }, new string[] { "relu", "linear" }, nnLearningRate);
        //    targetNetwork = new NeuralNetwork(predNetwork);
        //}

        //public Matrix<double> arrayToMatrix(double[] input)
        //{
        //    Matrix<double> output = Matrix<double>.Build.Dense(1, input.Length);
        //    for (int i = 0; i < input.Length; i++)
        //    {
        //        output[0, i] = input[i];
        //    }
        //    return output;
        //}
        //public int SelectAction(double[] state)
        //{
        //    int a;
        //    double randNum = rnd.NextDouble();
        //    if (randNum < epsilon)
        //    {
        //        a = rnd.Next(0, 8);
        //    }
        //    else
        //    {
        //        Matrix<double> Q = predNetwork.predict(arrayToMatrix(state));
        //        List<double> QList = Q.Row(0).ToList();
        //        a = QList.IndexOf(QList.Max());
        //    }
        //    return a;
        //}

        //public void Update(int prevAction, double[] prevState, double[] newState, double reward)
        //{
        //    current_steps++;
        //    Matrix<double> prevStateMatrix = arrayToMatrix(prevState);
        //    Matrix<double> newStateMatrix = arrayToMatrix(newState);

        //    Matrix<double> Q = predNetwork.predict(prevStateMatrix);

        //    Console.WriteLine(Q.ToString());

        //    if (current_steps == c)
        //    {
        //        targetNetwork.copyWeights(predNetwork);
        //        current_steps = 0;
        //    }

        //    double target = reward + gamma * targetNetwork.predict(newStateMatrix).Row(0).Max();
        //    Q[0, prevAction] = target;
        //    double error = predNetwork.train(prevStateMatrix, Q);
        //    Console.WriteLine("Model erorr: " + error);
        //}
    }
}
