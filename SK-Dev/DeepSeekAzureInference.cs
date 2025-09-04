using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.SemanticKernel.Connectors.AzureAIInference;

namespace SK_Dev
{
    internal class DeepSeekAzureInference
    {
        static async Task Main(string[] args)
        {

            // Build and get configuration from appsettings.json, environment variables, and user secrets
            IConfiguration config = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
                .AddEnvironmentVariables()
                .AddUserSecrets<DeepSeekAzureInference>()
                .Build();

            // Retrieve settings from configuration

            string? modelId = config["modelid"];
            string? endPoint = config["endpoint"];
            string? apikey = config["apiKey"];

            //Create a kernel builder and add Azure OpenAI chat completion service
            var builder = Kernel.CreateBuilder();
            builder.AddAzureAIInferenceChatCompletion(config["Inference:modelid"]!, config["Inference:apikey"],new Uri(config["Inference:endpoint"]!));

            //Build the kernel
            Kernel kernel = builder.Build();

            //Create chat history
            var history = new ChatHistory(systemMessage:"Talk very very rudely");

            //get reference to chat compilation service
            var chatCompleationService = kernel.GetRequiredService<IChatCompletionService>();


            //Prompt settings
            AzureAIInferencePromptExecutionSettings settings = new()
            {
                Temperature = 1f,
                MaxTokens=1500,
            };
            //

            //Managing chat history
            var reducer = new ChatHistoryTruncationReducer(targetCount: 2); //targetCount specifies how many messages should be there after reducer function gets trigger
            //var reducer = new ChatHistorySummarizationReducer(chatCompleationService, 2, 2);//2nd arg is targetCount,3rd arg + 2nd arg will trigger this reducer function
            //

            foreach (var attr in chatCompleationService.Attributes)
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
                Azure.AI.Inference.CompletionsUsage usage = null;
                await foreach (StreamingChatMessageContent responseChunk in chatCompleationService.GetStreamingChatMessageContentsAsync(history, settings))
                {
                    //Print response to console
                    Console.Write(responseChunk.Content);
                    fullMessage += responseChunk.Content;
                    usage = ((Azure.AI.Inference.StreamingChatCompletionsUpdate)responseChunk.InnerContent!).Usage;
                }

                //add response to chat history
                history.AddAssistantMessage(fullMessage);

                //Display number of tokens used. Model Specific
                Console.WriteLine($"\n\tInput Tokens: \t{usage?.PromptTokens}");
                Console.WriteLine($"\tOutput Tokens: \t{usage?.CompletionTokens}");
                Console.WriteLine($"\tTotal Tokens: \t{usage?.TotalTokens}");

                var reduceMessages = await reducer.ReduceAsync(history);
                if (reduceMessages != null)
                    history = new(reduceMessages);
            }

        }
    }
}

