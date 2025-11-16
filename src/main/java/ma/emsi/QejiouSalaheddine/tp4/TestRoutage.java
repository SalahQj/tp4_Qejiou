package ma.emsi.QejiouSalaheddine.tp4;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Classe principale pour le Test 3 du TP 4 : RAG avec Routage.
 */
public class TestRoutage {

    /**
     * Méthode utilitaire (suggérée par le TP) pour l'ingestion d'un document.
     * Prend un nom de ressource et un modèle d'embedding, et retourne un ContentRetriever prêt.
     */
    private static ContentRetriever createRetriever(String resourceName, EmbeddingModel embeddingModel) {
        System.out.println("Phase 1 : Ingestion de " + resourceName + "...");

        // 1. Charger le document
        ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();
        Document document;
        try (InputStream inputStream = TestRoutage.class.getResourceAsStream(resourceName)) {
            if (inputStream == null) {
                throw new RuntimeException("Erreur : Le fichier " + resourceName + " n'est pas trouvé.");
            }
            document = parser.parse(inputStream);
        } catch (Exception e) {
            throw new RuntimeException("Erreur lors du chargement de " + resourceName, e);
        }

        // 2. Découper le document
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 20);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("Document " + resourceName + " découpé en " + segments.size() + " segments.");

        // 3. Créer les embeddings
        Response<List<Embedding>> embeddingsResponse = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = embeddingsResponse.content();

        // 4. Stocker les embeddings
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        System.out.println("Embeddings pour " + resourceName + " stockés en mémoire.");

        // 5. Créer et retourner le ContentRetriever pour ce magasin
        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();
    }

    /**
     * Méthode pour configurer le logging (du Test 2)
     */
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) {
        // Activer le logging pour voir la décision du routeur
        configureLogger();

        // --- PHASE 1 : INGESTION (Optimisée) ---

        // 1. Créer UN SEUL modèle d'embedding local
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        System.out.println("Modèle d'embedding local chargé.");

        // 2. Créer 2 ContentRetrievers en utilisant la méthode utilitaire
        ContentRetriever ragRetriever = createRetriever("/rag.pdf", embeddingModel);
        ContentRetriever cuisineRetriever = createRetriever("/cuisine.txt", embeddingModel);


        // --- PHASE 2 : CONFIGURATION DU ROUTAGE ---
        System.out.println("Phase 2 : Préparation de l'assistant avec routage...");

        // 3. Créer le ChatModel (Gemini)
        String geminiApiKey = System.getenv("GEMINI_KEY");
        if (geminiApiKey == null || geminiApiKey.isEmpty()) {
            System.err.println("Erreur : GEMINI_KEY n'est pas définie.");
            return;
        }

        // ChatModel → ChatLanguageModel et suppression de logRequests/logResponses
        ChatLanguageModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiApiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .build();
        System.out.println("ChatModel Gemini chargé.");

        // 4. Créer la Map de descriptions pour le routeur
        Map<ContentRetriever, String> retrieverMap = new HashMap<>();
        retrieverMap.put(ragRetriever, "Informations sur l'IA, LangChain4j, et RAG (Retrieval-Augmented Generation)");
        retrieverMap.put(cuisineRetriever, "Recettes de cuisine, ingrédients, et techniques culinaires (sauce tomate, gâteau)");

        // 5. Créer le QueryRouter
        QueryRouter queryRouter = new LanguageModelQueryRouter(chatModel, retrieverMap);

        // 6. Créer le RetrievalAugmentor
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // 7. Créer l'Assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatModel)  // chatModel → chatLanguageModel
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        System.out.println("Assistant RAG avec Routage prêt. (tapez 'fin' pour quitter).");

        // 8. Lancer la boucle de conversation
        conversationAvec(assistant);
    }

    /**
     * Gère la boucle de conversation avec l'assistant.
     */
    private static void conversationAvec(Assistant assistant) {
        try (Scanner scanner = new Scanner(System.in)) {
            while (true) {
                System.out.println("==================================================");
                System.out.println("Posez votre question : ");
                String question = scanner.nextLine();
                if (question.isBlank()) {
                    continue;
                }
                if ("fin".equalsIgnoreCase(question)) {
                    System.out.println("Conversation terminée.");
                    break;
                }
                System.out.println("==================================================");
                String reponse = assistant.chat(question);
                System.out.println("Assistant : " + reponse);
            }
        }
    }
}