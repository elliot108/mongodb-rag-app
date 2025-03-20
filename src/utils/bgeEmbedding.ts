import { MongoDBAtlasVectorSearch, MongoDBAtlasVectorSearchLibArgs } from '@langchain/community/vectorstores/mongodb_atlas';
import { MongoClient } from "mongodb";
import { EmbeddingsInterface } from '@langchain/core/embeddings';
import { BGEEmbeddings } from './bge-large-en/BGEEmbeddings';
import axios from 'axios';
import dotenv from 'dotenv';
dotenv.config();

let embeddingsInstance: BGEEmbeddings | null = null;

const client = new MongoClient(process.env.MONGODB_URI!);
const namespace = "chatter.training_data";
const [dbName, collectionName] = namespace.split(".");
const collection = client.db(dbName).collection(collectionName);

// Custom class to handle BGE embeddings
// class BGEEmbeddings implements EmbeddingsInterface {
//     private apiUrl: string;

//     constructor(apiUrl: string) {
//         this.apiUrl = apiUrl;
//     }
//     async embedQuery(document: string): Promise<number[]> {
//         try {
//             const embeddings = await this.embedDocuments([document]);
//             return embeddings[0]; // Return the first (and only) embedding array
//         } catch (error) {
//             console.error("Error in embedQuery:", error);
//             throw error;
//         }
//     }

//     async embedDocuments(documents: string[]): Promise<number[][]> {
//         try {
//             const response = await axios.post(this.apiUrl, {
//                 inputs: documents,
//             }, {
//                 headers: {
//                     Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
//                 },
//             });

//             if (response.status === 200 && response.data) {
//                 return response.data; // Return all embedding arrays
//             } else {
//                 throw new Error(`Unexpected response from Hugging Face API: ${response.status}`);
//             }
//         } catch (error) {
//             console.error("Error fetching embeddings from Hugging Face API:", error);
//             throw error;
//         }
        
//     }

    
// }

export function getEmbeddingsTransformer(): BGEEmbeddings {
    try {
        // Ensure embeddingsInstance is initialized only once for efficiency
        if (!embeddingsInstance) {
            embeddingsInstance = new BGEEmbeddings("https://router.huggingface.co/hf-inference/models/BAAI/bge-large-en-v1.5");
        }

        return embeddingsInstance;
    } catch (error) {
        console.error("Error creating BGEEmbeddings instance:", error);
        throw new Error("Failed to create BGEEmbeddings instance.");
    }
}

export async function vectorStore(): Promise<MongoDBAtlasVectorSearch> {
    const vectorStore: MongoDBAtlasVectorSearch = new MongoDBAtlasVectorSearch(
        new BGEEmbeddings("https://router.huggingface.co/hf-inference/models/BAAI/bge-large-en-v1.5"),
        searchArgs()
    );
    return vectorStore;
}

export function searchArgs(): MongoDBAtlasVectorSearchLibArgs {
    const searchArgs: MongoDBAtlasVectorSearchLibArgs = {
        collection,
        indexName: "vector_index",
        textKey: "text",
        embeddingKey: "text_embedding",
    };
    return searchArgs;
}