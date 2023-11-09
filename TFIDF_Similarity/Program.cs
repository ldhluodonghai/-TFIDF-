﻿using JiebaNet.Segmenter;
using System;
using System.Linq;
using TFIDF_Model;

namespace TFIDF_Similarity
{
    class Program
    {
        static void Main(string[] args)
        {
            string[] documents =
           {
                "其居民楼旁有树木影响其采光，当时表示有美丽家园改造会综合考虑，但至今树木没有修剪还从其他地方移植了一棵树过来导致更加影响采光。",
                "原先的便民服务中心近期在进行装修施工，10月6日晚21:15左右仍在施工，噪音扰民情况严重。市民现诉求管理部门协调该处调整施工时间，夜间20:00之后，以及节假日不要进行施工。"
            };

            // Apply TF*IDF to the documents and get the resulting vectors.
            double[][] inputs = TFIDFModel.Transform(documents, 0);
            inputs = TFIDFModel.Normalize(inputs);
            //TFIDFModel.Save();
            // Display the output.
            for (int index = 0; index < inputs.Length; index++)
            {
                Console.WriteLine(documents[index]);

                foreach (double value in inputs[index])
                {
                    Console.Write(value + ", ");
                }

                Console.WriteLine("\n");
            }
            double[] vectorOne = inputs[0];
            double[] vectorTwo = inputs[1];

            double v = CalculateCosineSimilarity(vectorOne, vectorTwo);
            Console.WriteLine(v);
           /* string text = "我爱北,./;'京天安门";
            var segmenter = new JiebaSegmenter();
            var words = segmenter.Cut(text);
            string[] vs = TFIDF_Model.TFIDFModel.ChineseTokenize(text);
            foreach (var word in words)
            {
                Console.WriteLine(word);
            }
            Console.WriteLine(string.Join(" ", words));
            double[] x = { 1, 2, 3, 4, 5 };
            double[] y = { 2, 4, 6, 8, 10 };

            double correlation = CalculatePearsonCorrelation(x, y);
            Console.WriteLine($"皮尔逊相关系数: {correlation}");*/
        }

        /// <summary>
        /// 余弦相似度计算
        /// </summary>
        /// <param name="vectorOne"></param>
        /// <param name="vectorTwo"></param>
        /// <returns></returns>
        static double CalculateCosineSimilarity(double[] vectorOne, double[] vectorTwo)
        {
            //判断向量组维度是否相等
            if (vectorOne.Length != vectorTwo.Length)
            {
                throw new ArgumentException("Input vectors must have the same dimension");
            }

            double dotProduct = 0;
            double normVector1 = 0;
            double normVector2 = 0;

            //计算两向量相乘
            for (int i = 0; i < vectorOne.Length; i++)
            {
                dotProduct += vectorOne[i] * vectorTwo[i];
                normVector1 += vectorOne[i] * vectorOne[i];
                normVector2 += vectorTwo[i] * vectorTwo[i];
            }

            //判断是否为零向量
            if (normVector1 == 0 || normVector2 == 0)
            {
                return 0; // Handle division by zero
            }

            //返回余弦相识度
            return dotProduct / (Math.Sqrt(normVector1) * Math.Sqrt(normVector2));
        }

        /// <summary>
        /// 皮尔逊相关系数计算
        /// </summary>
        /// <param name="x">向量组</param>
        /// <param name="y">向量组</param>
        /// <returns></returns>
        static double CalculatePearsonCorrelation(double[] x, double[] y)
        {
            if (x.Length != y.Length)
            {
                throw new ArgumentException("输入数组的长度不一致");
            }

            int n = x.Length;

            // 计算x和y的平均值
            double avgX = x.Average();
            double avgY = y.Average();

            // 计算x和y的差值乘积之和
            double sumProduct = 0;
            for (int i = 0; i < n; i++)
            {
                sumProduct += (x[i] - avgX) * (y[i] - avgY);
            }

            // 计算x和y的差值的平方和
            double sumXSquare = 0;
            double sumYSquare = 0;
            for (int i = 0; i < n; i++)
            {
                sumXSquare += Math.Pow(x[i] - avgX, 2);
                sumYSquare += Math.Pow(y[i] - avgY, 2);
            }

            // 计算皮尔逊相关系数
            double correlation = sumProduct / Math.Sqrt(sumXSquare * sumYSquare);

            return correlation;
        }
    }
}
