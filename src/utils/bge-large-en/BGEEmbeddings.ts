import { EmbeddingsInterface } from '@langchain/core/embeddings';
import axios from 'axios';
import dotenv from 'dotenv';
dotenv.config();

export class BGEEmbeddings implements EmbeddingsInterface {
    private apiUrl: string;
    private apiKey: string;

    constructor(apiUrl: string = "https://router.huggingface.co/hf-inference/models/BAAI/bge-large-en-v1.5") {
        this.apiUrl = apiUrl;
        this.apiKey = process.env.HUGGINGFACE_API_KEY!;
        if (!this.apiKey) {
            throw new Error("HUGGINGFACE_API_KEY is not set in the environment variables.");
        }
    }

    /**
     * Embeds a single query string and returns its embedding as a number array.
     * @param document - The string to embed.
     * @returns A promise resolving to the embedding array.
     */
    async embedQuery(document: string): Promise<number[]> {
        try {
            const embeddings = await this.embedDocuments([document]);
            return embeddings[0]; // Return the first (and only) embedding array
        } catch (error) {
            console.error("Error in embedQuery:", error);
            throw error;
        }
    }

    /**
     * Embeds multiple documents and returns their embeddings as an array of number arrays.
     * @param documents - The array of strings to embed.
     * @returns A promise resolving to an array of embedding arrays.
     */
    async embedDocuments(documents: string[]): Promise<number[][]> {
        try {
            const response = await axios.post(this.apiUrl, {
                inputs: documents,
            }, {
                headers: {
                    Authorization: `Bearer ${this.apiKey}`,
                },
            });

            if (response.status === 200 && response.data) {
                return response.data; // Return all embedding arrays
            } else {
                throw new Error(`Unexpected response from Hugging Face API: ${response.status}`);
            }
        } catch (error) {
            console.error("Error fetching embeddings from Hugging Face API:", error);
            throw error;
        }
    }
}