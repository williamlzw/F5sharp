using NAudio.Wave.SampleProviders;
using NAudio.Wave;
using System.Text.RegularExpressions;
using System.Text;
using hyjiacan.py4n;
using JiebaNet.Segmenter;
using TorchSharp;

namespace F5Sharp
{
    public class WavHelpers
    {
        private static byte[] ConvertFloatToByte(float[] floatBuffer)
        {
            var byteBuffer = new byte[floatBuffer.Length * 4];
            Buffer.BlockCopy(floatBuffer, 0, byteBuffer, 0, byteBuffer.Length);
            return byteBuffer;
        }

        public static (float[] Samples, int SampleRate, int channels) ReadWav(string filePath)
        {
            using var reader = new WaveFileReader(filePath);
            var bytes = new byte[reader.Length];
            reader.Read(bytes, 0, bytes.Length);
            float[] samples;
            int sampleRate = reader.WaveFormat.SampleRate;
            var channels = reader.WaveFormat.Channels;
            if (reader.WaveFormat.BitsPerSample == 16)
            {
                samples = new float[bytes.Length / 2];
                for (int i = 0; i < samples.Length; i++)
                {
                    short int16 = BitConverter.ToInt16(bytes, i * 2);
                    samples[i] = int16 / 32768f;
                }
            }
            else if (reader.WaveFormat.BitsPerSample == 32 && reader.WaveFormat.Encoding == WaveFormatEncoding.IeeeFloat)
            {
                samples = new float[bytes.Length / 4];
                Buffer.BlockCopy(bytes, 0, samples, 0, bytes.Length);
            }
            else
            {
                throw new NotSupportedException("Unsupported WAV format.");
            }
            return (samples, sampleRate, channels);
        }
        public static void WriteWav(string filePath, short[] samples, int sampleRate, int channels)
        {
            // 1. 创建 16 位 PCM 格式的 WaveFormat
            var waveFormat = new WaveFormat(sampleRate, 16, channels);

            // 2. 将 short[] 直接转换为 byte[]（每个 short 占 2 字节）
            byte[] byteBuffer = new byte[samples.Length * 2];
            Buffer.BlockCopy(samples, 0, byteBuffer, 0, byteBuffer.Length);

            // 3. 写入 WAV 文件
            using (var writer = new WaveFileWriter(filePath, waveFormat))
            {
                writer.Write(byteBuffer, 0, byteBuffer.Length);
            }
        }

        public static void WriteWav(string filePath, float[] samples, int sampleRate, int channels)
        {
            // 1. 创建 WaveFormat（假设使用 32 位浮点数格式）
            var waveFormat = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, channels);

            // 2. 创建 WaveFileWriter 并写入数据
            using (var writer = new WaveFileWriter(filePath, waveFormat))
            {
                // 将 float[] 转换为 byte[]
                byte[] byteBuffer = new byte[samples.Length * 4]; // 每个 float 占 4 字节
                Buffer.BlockCopy(samples, 0, byteBuffer, 0, byteBuffer.Length);

                // 写入数据
                writer.Write(byteBuffer, 0, byteBuffer.Length);
            }
        }

        public static float[] ResampleAudio(float[] originalSamples, int originalSampleRate, int targetSampleRate, int channels = 1)
        {
            var originalFormat = WaveFormat.CreateIeeeFloatWaveFormat(originalSampleRate, channels);
            var provider = new RawSourceWaveStream(
                new MemoryStream(ConvertFloatToByte(originalSamples)),
                originalFormat
            ).ToSampleProvider();
            var resampler = new WdlResamplingSampleProvider(provider, targetSampleRate);
            var targetLength = (int)(originalSamples.Length * (double)targetSampleRate / originalSampleRate);
            var resampled = new float[targetLength];
            int samplesRead = resampler.Read(resampled, 0, resampled.Length);
            return resampled.Take(samplesRead).ToArray();
        }
    }

    public class TextChunker
    {
        public static List<string> ChunkText(string text, int maxChars = 135)
        {
            List<string> chunks = new List<string>();
            string currentChunk = string.Empty;

            // 使用正则表达式分割句子（支持中英文标点）
            string[] sentences = Regex.Split(
                text,
                @"(?<=[;:,.!?])\s+|(?<=[；：，。！？])"
            );

            foreach (string sentence in sentences)
            {
                if (string.IsNullOrEmpty(sentence)) continue;

                // 确定是否需要添加空格
                bool addSpace = sentence.Length > 0 &&
                                Encoding.UTF8.GetByteCount(new[] { sentence[^1] }) == 1;

                // 计算拼接后的新句子
                string newSegment = addSpace ? $"{sentence} " : sentence;

                // 计算字节长度（包含潜在的空格）
                int currentBytes = Encoding.UTF8.GetByteCount(currentChunk);
                int newBytes = Encoding.UTF8.GetByteCount(newSegment);

                // 判断是否超过限制（修正后的逻辑：包含空格的计算）
                if (currentBytes + newBytes <= maxChars)
                {
                    currentChunk += newSegment;
                }
                else
                {
                    if (!string.IsNullOrWhiteSpace(currentChunk))
                    {
                        chunks.Add(currentChunk.Trim());
                    }
                    currentChunk = newSegment;
                }
            }

            // 添加最后的剩余内容
            if (!string.IsNullOrEmpty(currentChunk))
            {
                chunks.Add(currentChunk.Trim());
            }

            return chunks;
        }
    }

    public class PinyinConverter
    {
        // 中文标点符号范围（扩展）
        private static bool IsChinesePunctuation(char c)
        {
            return (c >= '\u3000' && c <= '\u303F') ||  // 中文标点主范围
                   (c >= '\uFF00' && c <= '\uFFEF');    // 全角符号
        }

        public static List<List<string>> ConvertCharToPinyin(IEnumerable<string> textList)
        {
            var _segmenter = new JiebaSegmenter();
            var finalList = new List<List<string>>();

            foreach (var text in textList)
            {
                var charList = new List<string>();
                var processedText = PreprocessText(text);

                foreach (var seg in _segmenter.Cut(processedText))
                {
                    ProcessSegment(seg, charList);
                }

                finalList.Add(PostProcess(charList));
            }

            return finalList;
        }

        private static string PreprocessText(string text)
        {
            // 仅处理指定替换，保留中文标点
            return new StringBuilder(text)
                .Replace(';', ',')  // 只替换分号
                .ToString();
        }

        private static void ProcessSegment(string segment, List<string> charList)
        {
            if (IsPureAscii(segment))
            {
                HandleAsciiSegment(segment, charList);
            }
            else
            {
                HandleChineseSegment(segment, charList);
            }
        }

        private static bool IsPureAscii(string segment)
        {
            return segment.All(c => c < 256 && !IsChinesePunctuation(c));
        }

        private static void HandleAsciiSegment(string segment, List<string> charList)
        {
            foreach (var c in segment)
            {
                // 直接添加ASCII字符（含英文标点）
                charList.Add(c.ToString());
            }
        }

        private static void HandleChineseSegment(string segment, List<string> charList)
        {
            foreach (var c in segment)
            {
                if (IsChinese(c))
                {
                    var pinyin = GetPinyin(c);
                    AddChineseWithSpace(charList, pinyin);
                }
                else if (IsChinesePunctuation(c))
                {
                    // 直接添加中文标点
                    charList.Add(c.ToString());
                }
                else
                {
                    // 处理其他字符（如英文标点）
                    charList.Add(c.ToString());
                }
            }
        }

        private static bool IsChinese(char c)
        {
            return (c >= 0x4E00 && c <= 0x9FFF) ||
                   (c >= 0x3400 && c <= 0x4DBF);
        }

        private static string GetPinyin(char c)
        {
            try
            {
                var pinyinArray = Pinyin4Net.GetPinyin(c);
                return pinyinArray.FirstOrDefault()?.Replace("ü", "v") ?? c.ToString();
            }
            catch
            {
                return c.ToString();
            }
        }

        private static void AddChineseWithSpace(List<string> list, string pinyin)
        {
            // 当前列表非空 且 最后一个元素不是标点符号时添加空格
            if (list.Count > 0 && !IsPunctuation(list.Last()))
            {
                list.Add(" ");
            }
            list.Add(pinyin);
        }

        private static List<string> PostProcess(List<string> charList)
        {
            // 清理多余空格并处理特殊符号
            var result = new List<string>();
            string previous = null;

            foreach (var item in charList)
            {
                if (item == " ")
                {
                    if (previous != null && !IsPunctuation(previous))
                    {
                        result.Add(" ");
                    }
                }
                else
                {
                    result.Add(item);
                }
                previous = item;
            }

            return result.Where(s => !string.IsNullOrEmpty(s)).ToList();
        }

        private static bool IsPunctuation(string s)
        {
            return s.Length == 1 &&
                   (char.IsPunctuation(s[0]) || IsChinesePunctuation(s[0]));
        }
    }

    public class VocabLoader
    {
        public static (Dictionary<string, int> vocabCharMap, int vocabSize) LoadVocab(string vocabPath)
        {
            var vocabCharMap = new Dictionary<string, int>();
            int i = 0;

            using (var reader = new StreamReader(vocabPath, System.Text.Encoding.UTF8))
            {
                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    // 移除行末的换行符（类似 Python 的 char[:-1]）
                    string charKey = line.TrimEnd('\r', '\n');
                    vocabCharMap[charKey] = i;
                    i++;
                }
            }

            int vocabSize = vocabCharMap.Count;
            return (vocabCharMap, vocabSize);
        }
    }

    public class TextLengthCalculator
    {
        // 中文停顿标点符号（正则表达式模式）
        private static readonly string zhPausePuncPattern = @"[。，、；：？！]";

        /// <summary>
        /// 计算参考文本的加权长度（UTF-8字节长度 + 3 * 中文标点数量）
        /// </summary>
        public static int CalculateWeightedLength(string text)
        {
            if (string.IsNullOrEmpty(text))
                return 0;

            // 1. 计算UTF-8字节长度
            int utf8Length = Encoding.UTF8.GetByteCount(text);

            // 2. 统计中文标点数量
            int pausePuncCount = Regex.Matches(text, zhPausePuncPattern).Count;

            // 3. 加权计算总长度
            return utf8Length + 3 * pausePuncCount;
        }

        /// <summary>
        /// 同时计算参考文本和生成文本的加权长度
        /// </summary>
        public static (int refTextLen, int genTextLen) CalculateBothLengths(string refText, string genText)
        {
            int refLen = CalculateWeightedLength(refText);
            int genLen = CalculateWeightedLength(genText);
            return (refLen, genLen);
        }
    }

    public class TextUtils
    {
        /// <summary>
        /// 将字符串列表或嵌套字符串列表转换为索引张量，并进行填充。
        /// </summary>
        /// <param name="text">输入字符串列表或嵌套列表。</param>
        /// <param name="vocabCharMap">字符到索引的映射字典。</param>
        /// <param name="paddingValue">填充值（默认 -1）。</param>
        /// <returns>填充后的索引张量（形状：[batch_size, max_length]）。</returns>
        public static torch.Tensor ListStrToIdx(
            List<string> text,
            Dictionary<string, int> vocabCharMap,
            long paddingValue = -1
        )
        {
            // 1. 将每个字符串转换为索引列表
            List<torch.Tensor> listIdxTensors = new List<torch.Tensor>();
            foreach (var t in text)
            {
                // 获取每个字符的索引（若不存在则返回 0）
                var indices = t.Select(c =>
                    vocabCharMap.TryGetValue(c.ToString(), out int idx) ? idx : 0
                ).ToArray();

                // 创建 TorchSharp 张量（int32）
                var tensor = torch.tensor(indices, dtype: torch.int32);
                listIdxTensors.Add(tensor);
            }

            // 2. 使用 pad_sequence 进行填充
            var padded = torch.nn.utils.rnn.pad_sequence(listIdxTensors, batch_first: true, padding_value: paddingValue);
            return padded;
        }

        /// <summary>
        /// 重载方法，支持嵌套字符串列表（List<List<string>>）。
        /// </summary>
        public static torch.Tensor ListStrToIdx(
            List<List<string>> text,
            Dictionary<string, int> vocabCharMap,
            long paddingValue = -1
        )
        {
            List<torch.Tensor> listIdxTensors = new List<torch.Tensor>();
            foreach (var sublist in text)
            {
                var indices = sublist.Select(s =>
                    vocabCharMap.TryGetValue(s, out int idx) ? idx : 0
                ).ToArray();

                var tensor = torch.tensor(indices, dtype: torch.int32);
                listIdxTensors.Add(tensor);
            }

            var padded = torch.nn.utils.rnn.pad_sequence(listIdxTensors, batch_first: true, padding_value: paddingValue);
            return padded;
        }
    }
}