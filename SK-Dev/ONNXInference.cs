using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.HuggingFace;
using Microsoft.SemanticKernel.Connectors.Onnx;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SK_Dev
{
    internal class ONNXInference
    {
        static async Task Main(string[] args)
        {

            // Build and get configuration from appsettings.json, environment variables, and user secrets
            IConfiguration config = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
                .AddEnvironmentVariables()
                .AddUserSecrets<ONNXInference>()
                .Build();


            //Create a kernel builder and add Azure OpenAI chat completion service
            //var builder = Kernel.CreateBuilder();

            //ONNX
            //builder.AddOnnxRuntimeGenAIChatCompletion(config["ONNX:modelid"]!,config["ONNX:modelpath"]!);

            //Build the kernel
            //Kernel kernel = builder.Build();

            using OnnxRuntimeGenAIChatCompletionService chatCompletionService = new(config["ONNX:modelid"]!, config["ONNX:modelpath"]!);

            //Create chat history
            var history = new ChatHistory(systemMessage: "Talk very very rudely");

            //get reference to chat compilation service
            //var chatCompleationService = kernel.GetRequiredService<IChatCompletionService>();


            //Prompt settings
            OnnxRuntimeGenAIPromptExecutionSettings settings = new()
            {
                Temperature = 1f,
                MaxTokens = 1500,
            };
            //

            //Managing chat history
            var reducer = new ChatHistoryTruncationReducer(targetCount: 2); //targetCount specifies how many messages should be there after reducer function gets trigger
            //var reducer = new ChatHistorySummarizationReducer(chatCompleationService, 2, 2);//2nd arg is targetCount,3rd arg + 2nd arg will trigger this reducer function
            //

            foreach (var attr in chatCompletionService.Attributes)
                Console.WriteLine($"{attr.Key} \t\t{attr.Value}");

            while (true)
            {
                Console.Write("Enter your prompt:");
                var prompt = Console.ReadLine();
                if (string.IsNullOrEmpty(prompt))
                    break;


                //Get response from chat compilation service
                history.AddUserMessage(prompt);


                //Stream the output
                string fullMessage = "";
                await foreach (StreamingChatMessageContent responseChunk in chatCompletionService.GetStreamingChatMessageContentsAsync(history, settings))
                {
                    //Print response to console
                    Console.Write(responseChunk.Content);
                    fullMessage += responseChunk.Content;
                }

                //add response to chat history
                history.AddAssistantMessage(fullMessage);


                var reduceMessages = await reducer.ReduceAsync(history);
                if (reduceMessages != null)
                    history = new(reduceMessages);
            }

        }
    }
}
