using Annytab.Stemmer;
using JiebaNet.Segmenter;
using SotpWordsClassify;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;


namespace TFIDF_Model
{
    /// <summary>
    ///author:luodonghai
    /// </summary>
    public static class TFIDFModel
    {
        /// <summary>
        ///文档词汇表，包含每个词的IDF值。
        /// </summary>
        private static Dictionary<string, double> _vocabularyIDF = new Dictionary<string, double>();

        /// <summary>
        /// 将文档列表转换为相关的TF*IDF值。
        /// 如果一个词汇表还不存在，将根据文档中的词汇创建一个。
        /// </summary>
        /// <param name="documents">string[]</param>
        /// <param name="vocabularyThreshold">该术语在所有文档中出现的最小次数</param>
        /// <returns></returns>
        public static double[][] Transform(string[] documents, int vocabularyThreshold = 2)
        {
            List<List<string>> stemmedDocs;
            List<string> vocabulary;

            //同时得到词汇和文档。
            //vocabulary = GetVocabularyEng(documents, out stemmedDocs, vocabularyThreshold);
            vocabulary = GetVocabularyChina(documents, out stemmedDocs, vocabularyThreshold);

            if (_vocabularyIDF.Count == 0)
            {
                // 计算每个词汇表或语料库的IDF
                foreach (var term in vocabulary)
                {
                    double numberOfDocsContainingTerm = stemmedDocs.Where(d => d.Contains(term)).Count();
                    _vocabularyIDF[term] = Math.Log((double)stemmedDocs.Count / ((double)1 + numberOfDocsContainingTerm));
                }
            }

            // 将每个文档转换为tfidf值的矢量。
            return TransformToTFIDFVectors(stemmedDocs, _vocabularyIDF);
        }

        /// <summary>
        ///将派生文档列表(派生词列表)及其相关词汇表+ idf值转换为TF* idf值数组。
        /// </summary>
        /// <param name="stemmedDocs">字符串的列表</param>
        /// <param name="vocabularyIDF">双精度字符串字典(term/术语/名词, IDF)</param>
        /// <returns>double[][]</returns>
        private static double[][] TransformToTFIDFVectors(List<List<string>> stemmedDocs, Dictionary<string, double> vocabularyIDF)
        {
            // 将每个文档转换为tfidf值的矢量。
            List<List<double>> vectors = new List<List<double>>();
            foreach (var doc in stemmedDocs)
            {
                List<double> vector = new List<double>();

                foreach (var vocab in vocabularyIDF)
                {
                    // 术语频率=计算该术语在此文档中出现的次数。
                    double tf = doc.Where(d => d == vocab.Key).Count();
                    double tfidf = tf * vocab.Value;

                    vector.Add(tfidf);
                }

                vectors.Add(vector);
            }

            return vectors.Select(v => v.ToArray()).ToArray();
        }

        /// <summary>
        /// 使用L2-Norm归一化TF*IDF向量数组。
        /// Xi = Xi / Sqrt(X0^2 + X1^2 + .. + Xn^2)
        /// </summary>
        /// <param name="vectors">double[][]</param>
        /// <returns>double[][]</returns>
        public static double[][] Normalize(double[][] vectors)
        {
            // Normalize the vectors using L2-Norm.
            List<double[]> normalizedVectors = new List<double[]>();
            foreach (var vector in vectors)
            {
                var normalized = Normalize(vector);
                normalizedVectors.Add(normalized);
            }

            return normalizedVectors.ToArray();
        }

        /// <summary>
        /// 使用L2-Norm归一化TF*IDF向量。
        /// Xi = Xi / Sqrt(X0^2 + X1^2 + .. + Xn^2)
        /// </summary>
        /// <param name="vectors">double[][]</param>
        /// <returns>double[][]</returns>
        public static double[] Normalize(double[] vector)
        {
            List<double> result = new List<double>();

            double sumSquared = 0;
            foreach (var value in vector)
            {
                sumSquared += value * value;
            }

            double SqrtSumSquared = Math.Sqrt(sumSquared);

            foreach (var value in vector)
            {
                // L2-norm: Xi = Xi / Sqrt(X0^2 + X1^2 + .. + Xn^2)
                result.Add(value / SqrtSumSquared);
            }

            return result.ToArray();
        }

        /// <summary>
        /// 将TFIDF词汇表保存到磁盘。
        /// </summary>
        /// <param name="filePath">File path</param>
        public static void Save(string filePath = "vocabulary.dat")
        {
            // 保存结果到磁盘
            using (FileStream fs = new FileStream(filePath, FileMode.Create))
            {
                BinaryFormatter formatter = new BinaryFormatter();
                formatter.Serialize(fs, _vocabularyIDF);
            }
        }

        /// <summary>
        /// 从磁盘加载TFIDF词汇表
        /// </summary>
        /// <param name="filePath">File path</param>
        public static void Load(string filePath = "vocabulary.dat")
        {
            //从磁盘加载
            using (FileStream fs = new FileStream(filePath, FileMode.Open))
            {
                BinaryFormatter formatter = new BinaryFormatter();
                _vocabularyIDF = (Dictionary<string, double>)formatter.Deserialize(fs);
            }
        }

        /// <summary>
        /// 解析和标记文档列表，返回词汇表。
        /// </summary>
        /// <param name="docs">string[]</param>
        /// <param name="stemmedDocs">字符串列表的列表</param>
        /// <returns>词汇的字符串的列表</returns>
        private static List<string> GetVocabularyEng(string[] docs, out List<List<string>> stemmedDocs, int vocabularyThreshold)
        {
            List<string> vocabulary = new List<string>();
            Dictionary<string, int> wordCountList = new Dictionary<string, int>();
            stemmedDocs = new List<List<string>>();

            int docIndex = 0;

            foreach (var doc in docs)
            {
                List<string> stemmedDoc = new List<string>();

                docIndex++;

                if (docIndex % 100 == 0)
                {
                    Console.WriteLine("Processing " + docIndex + "/" + docs.Length);
                }

                string[] parts2 = Tokenize(doc);

                List<string> words = new List<string>();
                foreach (string part in parts2)
                {
                    // 消除非词汇的数字字符。
                    string stripped = Regex.Replace(part, "[^a-zA-Z0-9]", "");

                    if (!EnglishStopWords.stopWordsList.Contains(stripped.ToLower()))
                    {
                        try
                        {
                            //var english = new EnglishWord(stripped);
                            var english = new EnglishStemmer();
                            string stem = english.GetSteamWord(stripped);
                            //string stem = english.Stem;
                            words.Add(stem);

                            if (stem.Length > 0)
                            {
                                // 建立单词计数表。
                                if (wordCountList.ContainsKey(stem))
                                {
                                    wordCountList[stem]++;
                                }
                                else
                                {
                                    wordCountList.Add(stem, 0);
                                }

                                stemmedDoc.Add(stem);
                            }
                        }
                        catch
                        {
                        }
                    }
                }

                stemmedDocs.Add(stemmedDoc);
            }

            // 找到最热门的单词。
            var vocabList = wordCountList.Where(w => w.Value >= vocabularyThreshold);
            foreach (var item in vocabList)
            {
                vocabulary.Add(item.Key);
            }

            return vocabulary;
        }

        /// <summary>
        /// 中文解析和标记文档列表，返回词汇表。
        /// </summary>
        /// <param name="docs">string[]</param>
        /// <param name="stemmedDocs">字符串列表的列表</param>
        /// <returns>词汇的字符串的列表</returns>
        private static List<string> GetVocabularyChina(string[] docs, out List<List<string>> stemmedDocs, int vocabularyThreshold)
        {
            List<string> vocabulary = new List<string>();
            Dictionary<string, int> wordCountList = new Dictionary<string, int>();
            stemmedDocs = new List<List<string>>();

            int docIndex = 0;

            foreach (var doc in docs)
            {
                List<string> stemmedDoc = new List<string>();

                docIndex++;

                if (docIndex % 100 == 0)
                {
                    Console.WriteLine("Processing " + docIndex + "/" + docs.Length);
                }

                string[] parts2 = ChineseTokenize(doc);

                List<string> words = new List<string>();
                foreach (string part in parts2)
                {
                    // 消除非词汇的数字字符。
                    //string stripped = Regex.Replace(part, "[^a-zA-Z0-9]", "");

                    if (!ChineseStopWords.stopWordsList.Contains(part))
                    {
                        try
                        {
                            //var english = new EnglishWord(stripped);
                            //new ChinaStemmer();
                            //var english = new EnglishStemmer();
                            string stem = part.Trim();
                            //string stem = english.Stem;
                            words.Add(stem);

                            if (stem.Length > 0)
                            {
                                // 建立单词计数表。
                                if (wordCountList.ContainsKey(stem))
                                {
                                    wordCountList[stem]++;
                                }
                                else
                                {
                                    wordCountList.Add(stem, 0);
                                }

                                stemmedDoc.Add(stem);
                            }
                        }
                        catch
                        {
                        }
                    }
                }

                stemmedDocs.Add(stemmedDoc);
            }

            // 找到最热门的单词。
            var vocabList = wordCountList.Where(w => w.Value >= vocabularyThreshold);
            foreach (var item in vocabList)
            {
                vocabulary.Add(item.Key);
            }

            return vocabulary;

        }
        /// <summary>
        /// English,标记一个字符串，返回它的单词列表。中文不行
        /// </summary>
        /// <param name="text">string</param>
        /// <returns>string[]</returns>
        
        private static string[] Tokenize(string text)
        {
            // Strip all HTML.
            text = Regex.Replace(text, "<[^<>]+>", "");

            // Strip numbers.
            text = Regex.Replace(text, "[0-9]+", "number");

            // Strip urls.
            text = Regex.Replace(text, @"(http|https)://[^\s]*", "httpaddr");

            // Strip email addresses.
            text = Regex.Replace(text, @"[^\s]+@[^\s]+", "emailaddr");

            // Strip dollar sign.
            text = Regex.Replace(text, "[$]+", "dollar");

            // Strip usernames.
            text = Regex.Replace(text, @"@[^\s]+", "username");

            // Tokenize and also get rid of any punctuation
            return text.Split(" @$/#.-:&*+=[]?!(){},''\">_<;%\\".ToCharArray());
        }

        /// <summary>
        /// 中文分词器
        /// </summary>
        /// <param name="text"></param>
        /// <returns></returns>
        public static string[] ChineseTokenize(string text)
        {
            // 创建正则表达式模式，用于匹配标点符号
            string pattern = @"[^\w\s]";

            // 使用空字符串替换所有匹配的标点符号
            string result = Regex.Replace(text, pattern, "");
            string v = Regex.Replace(result, @"[A-Za-z]", "");
            var segmenter = new JiebaSegmenter();
            IEnumerable<string> enumerable = segmenter.Cut(v);
            
            string[] vs = enumerable.ToArray();
            return vs;
        }
    }
}
