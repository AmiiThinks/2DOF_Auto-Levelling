using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using System.IO;

namespace MLCSharp
{
    
    public class RLAgent
    {
        //Q Learning
        int numStates { get; set; }
        int numActions { get; set; }
        public double gamma { get; private set; } //discount factor
        public double epsilon { get; private set; } //exploration rate
        public int statePropAmount { get; set; } //Number of nearby states to propagate a state update
        public double statePropFactor { get; set; } //Amount to propagate a state update
        public Matrix<double> Q { get; set; }
        public double alpha { get; set; }
        public Random rnd { get; set; }

        private int prevState;
        private int prevAction;

        public List<int> statesVisited;

        public RLAgent(int numStates, int numActions, double gamma, double epsilon, double alpha, int statePropAmount, double statePropFactor, string filePath = "")
        {
            this.numStates = numStates;
            this.numActions = numActions;
            this.gamma = gamma;
            this.epsilon = epsilon;
            this.alpha = alpha;
            this.statePropAmount = statePropAmount;
            this.statePropFactor = statePropFactor;
            prevState = -1;
            rnd = new Random();
            if (filePath != string.Empty && File.Exists(filePath))
            {
                Q = LoadQ(filePath);
            }
            else
            {
                Console.WriteLine("Create new Q table");
                Q = Matrix<double>.Build.Dense(numStates, numActions, -90);
            }
            statesVisited = new List<int>();
        }

        public int SelectAction(int state)
        {
            int a;
            double randNum = rnd.NextDouble();
            if (randNum < epsilon)
            {
                a = rnd.Next(0, numActions);
            }
            else
            {
                var QRow = Q.Row(state).ToList();
                a = QRow.IndexOf(QRow.Max());
            }
            prevAction = a;
            return a - numActions / 2;
        }

        public void SetAction(int action)
        {
            prevAction = Math.Min(Math.Max(action + numActions / 2, 0), numActions - 1);
        }

        public double Update(int newState, double reward)
        {
            double tdError = 0;
            if (prevState != -1)
            {
                double target = reward + gamma * Q.Row(newState).Max();
                double update = (1 - alpha) * Q[prevState, prevAction] + alpha * target;
                Q[prevState, prevAction] = update;
                for (int i = 1; i <= statePropAmount; i++)
                {
                    double propagatedFactor = Math.Pow(statePropFactor, i);
                    if (prevState + i < numStates)
                    {
                        if(!(prevState <= numStates/2 && prevState + i > numStates/2))
                        {
                            Q[prevState + i, prevAction] = (1 - propagatedFactor) * Q[prevState + i, prevAction] + propagatedFactor * update;
                        }

                    }
                    if (prevState - i >= 0)
                    {
                        if (!(prevState >= numStates / 2 && prevState - i < numStates / 2))
                        {
                            Q[prevState - i, prevAction] = (1 - propagatedFactor) * Q[prevState, prevAction] + propagatedFactor * update;
                        }
                    }
                }

                tdError = Q[prevState, prevAction] - target;

            }
            prevState = newState;
            if (!statesVisited.Exists(x => x == prevState))
            {
                statesVisited.Add(prevState);
            }
            Console.WriteLine(String.Format("Visited {0}/{1}", statesVisited.Count, numStates));
            return tdError;
        }

        public void SaveQ(string filePath)
        {
            WriteToBinaryFile<Matrix<Double>>(filePath, Q);
        }

        public Matrix<double> LoadQ(string filePath)
        {
            return ReadFromBinaryFile<Matrix<double>>(filePath);
        }

        //From https://stackoverflow.com/a/22417240
        // <summary>
        /// Writes the given object instance to a binary file.
        /// <para>Object type (and all child types) must be decorated with the [Serializable] attribute.</para>
        /// <para>To prevent a variable from being serialized, decorate it with the [NonSerialized] attribute; cannot be applied to properties.</para>
        /// </summary>
        /// <typeparam name="T">The type of object being written to the binary file.</typeparam>
        /// <param name="filePath">The file path to write the object instance to.</param>
        /// <param name="objectToWrite">The object instance to write to the binary file.</param>
        /// <param name="append">If false the file will be overwritten if it already exists. If true the contents will be appended to the file.</param>
        private void WriteToBinaryFile<T>(string filePath, T objectToWrite, bool append = false)
        {
            using (Stream stream = File.Open(filePath, append ? FileMode.Append : FileMode.Create))
            {
                var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                binaryFormatter.Serialize(stream, objectToWrite);
            }
        }

        /// <summary>
        /// Reads an object instance from a binary file.
        /// </summary>
        /// <typeparam name="T">The type of object to read from the binary file.</typeparam>
        /// <param name="filePath">The file path to read the object instance from.</param>
        /// <returns>Returns a new instance of the object read from the binary file.</returns>
        private T ReadFromBinaryFile<T>(string filePath)
        {
            using (Stream stream = File.Open(filePath, FileMode.Open))
            {
                var binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                return (T)binaryFormatter.Deserialize(stream);
            }
        }
    }
}
