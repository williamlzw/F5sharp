
using F5Sharp;
using TorchSharp;

public class Program
{
    public static void Test()
    {
        string vocabFile = "model\\vocab.txt";
        string audioFile = "model\\trump_0.wav";
        string refText = "In short, we embarked on a mission to make America great again for all Americans.";
        string genText = "简而言之, 我们开启了一项使命, 要让美国再次伟大, 造福所有美国人。";
        string resultFile = "out.wav";
        var tool = new F5Tool(new ModelConfig
        {
            ModelPathPreprocess = "model\\F5_Preprocess.onnx",
            ModelPathTransformer = "model\\F5_Transformer.onnx",
            ModelPathDecode = "model\\F5_Decode.onnx"
        });
        tool.Inference(vocabFile, audioFile, refText, genText, resultFile);

    }

    public static void Main()
    {
        Test();
    }
}

