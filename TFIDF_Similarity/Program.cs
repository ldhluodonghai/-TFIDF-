using JiebaNet.Segmenter;
using System;
using TFIDF_Model;

namespace TFIDF_Similarity
{
    class Program
    {
        static void Main(string[] args)
        {
            string[] documents =
           {
                "北京天安门dgs",
                "北京天安门发到这个f我认为特vsfsf中的发个地址发给的发展郭德纲"
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
            string text = "我爱北,./;'京天安门";
            var segmenter = new JiebaSegmenter();
            var words = segmenter.Cut(text);
            string[] vs = TFIDF_Model.TFIDFModel.ChineseTokenize(text);
            foreach (var word in words)
            {
                Console.WriteLine(word);
            }
            Console.WriteLine(string.Join(" ", words));
        }
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
    }
}
