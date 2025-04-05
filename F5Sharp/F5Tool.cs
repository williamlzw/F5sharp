using Microsoft.ML.OnnxRuntime;
using System;
using System.Text.RegularExpressions;
using System.Text;
using TorchSharp;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace F5Sharp
{
    public class ModelConfig
    {
        public string ModelPathPreprocess { get; set; }
        public string ModelPathTransformer { get; set; }
        public string ModelPathDecode { get; set; }
        public string CmvnPath { get; set; }
    }
    public class F5Tool
    {
        private readonly InferenceSession _sessionA;
        private readonly InferenceSession _sessionB;
        private readonly InferenceSession _sessionC;

        public F5Tool(ModelConfig config)
        {
            var options = new SessionOptions();
            options.AddSessionConfigEntry("session.intra_op.allow_spinning", "1");
            options.AddSessionConfigEntry("session.inter_op.allow_spinning", "1");
            options.AddSessionConfigEntry("session.set_denormal_as_zero", "1");
            options.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL;
            options.LogVerbosityLevel = 4;
            options.EnableCpuMemArena = true;
            options.ExecutionMode = ExecutionMode.ORT_SEQUENTIAL;
            options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            options.AppendExecutionProvider_CUDA();
            _sessionA = new InferenceSession(config.ModelPathPreprocess, options);
            _sessionB = new InferenceSession(config.ModelPathTransformer, options);
            _sessionC = new InferenceSession(config.ModelPathDecode, options);
        }

        public void Inference(string vocabFile, string audioFile, string refText, string genText, string resultFile)
        {
            var (vocabMap, vocabSize) = VocabLoader.LoadVocab(vocabFile);
            var (waveform, audioLength) = LoadAudio(audioFile);
            int hopLength = 256;
            var refAudioLength = audioLength / hopLength + 1;
            var (refLen, genLen) = TextLengthCalculator.CalculateBothLengths(refText, genText);
           
            long maxDurationValue = refAudioLength + (int)refAudioLength / refLen * genLen;
            var maxDurationValueTensor = torch.tensor(maxDurationValue, dtype: torch.int64);

            var inputTexts = new List<string>
            {
                refText + genText
            };
            var textLists = PinyinConverter.ConvertCharToPinyin(inputTexts);
            var textIdsTensor = TextUtils.ListStrToIdx(textLists, vocabMap);
            var audio = OrtValue.CreateTensorValueFromMemory(
                waveform.data<short>().ToArray(), new long[] { 1, 1, waveform.shape[0] });
            var textIds = OrtValue.CreateTensorValueFromMemory(
                textIdsTensor.data<int>().ToArray(), new long[] { 1, textIdsTensor.shape[1] });
            var maxDuration = OrtValue.CreateTensorValueFromMemory(
                maxDurationValueTensor.data<Int64>().ToArray(), new long[] { 1 });

            var inputsValuesSessionA = new List<OrtValue>
             {
                 audio,
                 textIds,
                 maxDuration 
             };

            var resultsSessionA = _sessionA.Run(new RunOptions(), _sessionA.InputNames, inputsValuesSessionA, _sessionA.OutputNames);
            var noise = resultsSessionA[0];
            var ropeCos = resultsSessionA[1];
            var ropeSin = resultsSessionA[2];
            var catMelText = resultsSessionA[3];
            var catMelTextDrop = resultsSessionA[4];
            var refSignalLen = resultsSessionA[5];
            var timeSteps = OrtValue.CreateTensorValueFromMemory(
                new int[] {0}, new long[] { 1 });
            var inputsValuesSessionB = new List<OrtValue>
             {
                 noise,
                 ropeCos,
                 ropeSin,
                 catMelText,
                 catMelTextDrop,
                 timeSteps
             };
            IDisposableReadOnlyCollection<OrtValue> resultsSessionB = null;
            for (int i = 0; i < 32; i++)
            {
                Console.WriteLine($"step:{i}");
                (resultsSessionB) = _sessionB.Run(new RunOptions(), _sessionB.InputNames, inputsValuesSessionB, _sessionB.OutputNames);
                noise = resultsSessionB[0];
                timeSteps = resultsSessionB[1];
                inputsValuesSessionB = new List<OrtValue>
                 {
                     noise,
                     ropeCos,
                     ropeSin,
                     catMelText,
                     catMelTextDrop,
                     timeSteps
                 };
            }
            var inputsValuesSessionC = new List<OrtValue>
             {
                 noise,
                 refSignalLen
             };
            var resultsSessionC = _sessionC.Run(new RunOptions(), _sessionC.InputNames, inputsValuesSessionC, _sessionC.OutputNames);
            var generatedSignal = resultsSessionC[0].GetTensorDataAsSpan<Int16>().ToArray();
            WavHelpers.WriteWav(resultFile, generatedSignal, 24000, 1);
        }
        
        public (torch.Tensor, int) LoadAudio(string audioPath)
        {
            var (samples, sampleRate, channels) = WavHelpers.ReadWav(audioPath);
            if (sampleRate != 24000)
            {
                samples = WavHelpers.ResampleAudio(
                    originalSamples: samples,
                    originalSampleRate: sampleRate,
                    targetSampleRate: 24000,
                    channels: channels
                );
            }
            var audioLength = samples.Length / channels;
            var waveform = torch.tensor(samples).reshape(-1, channels).t().mean(new long[] { 0 }).reshape(-1);
            waveform = (waveform * 32767).to(torch.@short);
            return (waveform, audioLength);
        }
    }
}
