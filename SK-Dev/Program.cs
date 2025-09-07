using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using System.Threading.Tasks;

namespace SK_Dev
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            // Build and get configuration from appsettings.json, environment variables, and user secrets
            IConfiguration config = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
                .AddEnvironmentVariables()
                .AddUserSecrets<Program>()
                .Build();

            // Retrieve settings from configuration

            string? modelId = config["modelid"];
            string? endPoint = config["endpoint"];
            string? apikey = config["apiKey"];
            var builder=Kernel.CreateBuilder();
            builder.AddAzureOpenAIChatCompletion(modelId!, endPoint!, apikey!);

            Kernel kernel = builder.Build();

            var history = new ChatHistory();

            //get reference to chat compilation service
            var chatCompleationService=kernel.GetRequiredService<IChatCompletionService>();


            //Prompt settings
            OpenAIPromptExecutionSettings settings = new()
            {
                ChatSystemPrompt = "Talk very very rudely",
                Temperature=1,
                //MaxTokens=200,
            };
            //

            //Managing chat history
            //var reducer = new ChatHistoryTruncationReducer(targetCount: 2); //targetCount specifies how many messages should be there after reducer function gets trigger
            var reducer = new ChatHistorySummarizationReducer(chatCompleationService,2,2);//2nd arg is targetCount,3rd arg + 2nd arg will trigger this reducer function
            //

            while (true)
            {
                Console.Write("Enter your prompt:");
                var prompt = Console.ReadLine();
                if (string.IsNullOrEmpty(prompt))
                    break;


                //Get response from chat compilation service
                history.AddUserMessage(prompt);

                //Without stream
                //var response = await chatCompleationService.GetChatMessageContentAsync(history, settings);

                //output some of the parameters
                // Console.WriteLine(response.Content);
                ////add response to chat history
                //history.Add(response);
                //OpenAI.Chat.ChatTokenUsage usage = ((OpenAI.Chat.ChatCompletion)response.InnerContent).Usage;


                //Stream the output
                string fullMessage = "";
                OpenAI.Chat.ChatTokenUsage usage = null;
                await foreach(StreamingChatMessageContent responseChunk in chatCompleationService.GetStreamingChatMessageContentsAsync(history,settings))
                {
                    //Print response to console
                    Console.Write(responseChunk.Content);
                    fullMessage += responseChunk.Content;
                    usage = ((OpenAI.Chat.StreamingChatCompletionUpdate)responseChunk.InnerContent).Usage;
                }

                //add response to chat history
                history.AddAssistantMessage(fullMessage);

                //Display number of tokens used. Model Specific
                Console.WriteLine($"\n\tInput Tokens: \t{usage?.InputTokenCount}");
                Console.WriteLine($"\tOutput Tokens: \t{usage?.OutputTokenCount}");
                Console.WriteLine($"\tTotal Tokens: \t{usage?.TotalTokenCount}");

                var reduceMessages=await reducer.ReduceAsync(history);
                if(reduceMessages != null)  
                    history=new(reduceMessages);
            }

        }
    }
}
