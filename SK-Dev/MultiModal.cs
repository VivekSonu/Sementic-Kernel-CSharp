using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.AzureOpenAI;
using Microsoft.SemanticKernel.Connectors.HuggingFace;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace SK_Dev
{
    internal class MultiModal
    {
        static async Task Main(string[] args)
        {
            // Build and get configuration from appsettings.json, environment variables, and user secrets
            IConfiguration config = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
                .AddEnvironmentVariables()
                .AddUserSecrets<MultiModal>()
                .Build();

            //Initialize the Azure OpenAI Chat Completion Service with necessary parameters
            //AzureOpenAIChatCompletionService chatCompletionService = new(
            //    deploymentName: config["DEPLOYMENT_NAME"]!,
            //    apiKey: config["API_KEY"]!,
            //    endpoint: config["ENDPOINT"]!,
            //    modelId: config["MODEL_ID"]!
            // );

            //Create a kernel builder and add Azure OpenAI chat completion service
            var builder = Kernel.CreateBuilder();

            //hugging face
            builder.AddHuggingFaceChatCompletion(config["HuggingFace:modelid"]!, new Uri(config["HuggingFace:endpoint"]!), config["HuggingFace:apikey"]!);

            //Build the kernel
            Kernel kernel = builder.Build();

            //get reference to chat compilation service
            var chatCompletionService = kernel.GetRequiredService<IChatCompletionService>();

            //var executionSettings = new HuggingFacePromptExecutionSettings
            //{
            //    ResponseFormat=typeof(CameraResult)
            //};

            //Get all image files from "images" directory
            var imageFiles = Directory.GetFiles(Path.Combine(Directory.GetCurrentDirectory(),"images"), "*.jpg");
            foreach (var imageFile in imageFiles)
            {
                //Load Image into memory
                Console.WriteLine($"Image:{imageFile}");
                byte[] bytes = File.ReadAllBytes(imageFile);

                //Create a chat history with an initial system message
                ChatHistory history = new ChatHistory(
                     @"you are a traffic analyzer AI that monitors traffic congestion images and congestion level.
                     Heavy congestion level is when there is very little room between cars and vehicles are braking.
                     Medium congestion is when there is a lot of cars but they are not braking.
                     Low traffic is when there are few cars on the road.
                     In addition, attempt to determine if the image was taken with a malfunctioning camera by looking for distorted image or missing content.
                     Can you put content in a JSON object with the following schema:
                     {
                      IsBroken ,
                      TrafficCongestionLevel TrafficCongestion ,
                      Analysis ,
                     }
                     Return ONLY valid JSON. No text, no explanation.
                     "
                    );

                // Add user messages to the chat history
                history.AddUserMessage(
                    [
                        new ImageContent(bytes, "image/jpeg"),
                        new TextContent("Analyze the image and determine the traffic congestion level. Also determine if the camera is malfunctioning")
                    ]
                 );

                //Get the chat message content from the chat completion service
                var response = await chatCompletionService.GetChatMessageContentAsync(chatHistory: history);

                //Console.WriteLine(response.Content);
                //Console.WriteLine(new string('-', 40));

                // Get the model’s raw text
                var resultText = response.Content.Replace("```json", "").Trim().Replace("```", "").Trim();

                //Console.WriteLine("Raw model output:");
                //Console.WriteLine(resultText);

                // Try to deserialize JSON into your class
                try
                {
                    var options= new JsonSerializerOptions
                    {
                        Converters = { new System.Text.Json.Serialization.JsonStringEnumConverter() }
                    };
                    var cameraResult = JsonSerializer.Deserialize<CameraResult>(resultText, options);

                    if (cameraResult != null)
                    {
                        Console.WriteLine("\nParsed object:");
                        Console.WriteLine($"IsBroken: {cameraResult.IsBroken}");
                        Console.WriteLine($"TrafficCongestion: {cameraResult.TrafficCongestion}");
                        Console.WriteLine($"Analysis: {cameraResult.Analysis}");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"❌ Failed to parse JSON: {ex.Message}");
                }

                //artificial delay
                await Task.Delay(2000);
            }
        }
    }

    public class CameraResult
    {
        public bool IsBroken { get; set; }

        public TrafficCongestionLevel TrafficCongestion { get; set; }
        public string Analysis { get; set; }
    }

    public enum TrafficCongestionLevel
    {
        Low,
        Moderate,
        Heavy,
        Unknown
    }
}


//{
//    "HuggingFace": {
//        //"modelid": "deepseek-ai/DeepSeek-R1:hyperbolic",
//        "modelid": "microsoft/phi-4:nebius",
//    "apikey": "",
//    "endpoint": "https://router.huggingface.co"
//    },
//  "ONNX": {
//        "modelid": "phi3",
//    "modelpath": "D:\\New folder\\OneDrive - vit.ac.in\\Desktop\\Models\\Phi-3-mini-4k-instruct-onnx\\cpu_and_mobile\\cpu-int4-rtn-block-32-acc-level-4"
//  }
//}
