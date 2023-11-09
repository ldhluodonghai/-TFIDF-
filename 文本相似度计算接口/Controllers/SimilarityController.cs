using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using TFIDF_Model;

namespace 文本相似度计算接口.Controllers
{
    [Route("api/[controller]/[action]")]
    [ApiController]
    public class SimilarityController : ControllerBase
    {
        [HttpPost]
        public  double GetSimilarityValue(string text1,string text2)
        {
            string[] documents =
           {
                text1,text2
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
            return v;
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
