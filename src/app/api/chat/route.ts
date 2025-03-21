import { StreamingTextResponse, LangChainStream, Message } from 'ai';
// import { ChatDeepSeek } from '@langchain/community/chat_models/deepseek';
import { ChatOpenAI } from 'langchain/chat_models/openai'; // Commented out OpenAI import

import { ConversationalRetrievalQAChain } from 'langchain/chains';
// import { vectorStore } from '@/utils/openai'; // Commented out OpenAI vectorStore import
import { vectorStore } from '@/utils/bgeEmbedding'; // Using DeepSeek vectorStore
import { NextResponse } from 'next/server';
import { BufferMemory } from "langchain/memory";

export async function POST(req: Request) {
    try {
        const { stream, handlers } = LangChainStream();
        const body = await req.json();
        const messages: Message[] = body.messages ?? [];
        const question = messages[messages.length - 1].content;

        // OpenAI model (commented out)
        /*
        const model = new ChatOpenAI({
            temperature: 0.8,
            streaming: true,
            callbacks: [handlers],
        });
        */
        // DeepSeek model
        const model = new ChatOpenAI({
            modelName: "deepseek-chat",
            temperature: 0.8,
            streaming: true,
            callbacks: [handlers],
        });

        const store = await vectorStore();
        const retriever = store.asRetriever({ 
            "searchType": "mmr", 
            "searchKwargs": { "fetchK": 10, "lambda": 0.25 } 
        });

        const conversationChain = ConversationalRetrievalQAChain.fromLLM(model, retriever, {
            memory: new BufferMemory({
                memoryKey: "chat_history",
            }),
        });

        await conversationChain.invoke({
            "question": question,
        });

        return new StreamingTextResponse(stream);
    }
    catch (e) {
        return NextResponse.json({ message: 'Error Processing' }, { status: 500 });
    }
}