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
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.io.InputStream;
import java.util.List;
import java.util.Scanner;

// NOUVEAU (Test 2) : Imports pour le logging détaillé
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Classe principale pour le Test 1 & 2 du TP 4 : RAG "Naïf" décomposé avec Logging.
 */
public class RagNaif {

    // NOUVEAU (Test 2) : Méthode pour configurer le logging
    private static void configureLogger() {
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // Ajuster niveau
        // Ajouter un handler pour la console pour faire afficher les logs
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) {

        // NOUVEAU (Test 2) : Activer le logger
        configureLogger();

        System.out.println("Début du RAG Naïf (Test 2 avec Logging)...");

        // --- PHASE 1 : INGESTION (Chargement des connaissances) ---
        System.out.println("Phase 1 : Ingestion du document PDF...");

        // 1. Charger le document PDF depuis src/main/resources
        ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();
        Document document;
        try (InputStream inputStream = RagNaif.class.getResourceAsStream("/rag.pdf")) {
            if (inputStream == null) {
                System.err.println("Erreur : Le fichier rag.pdf n'est pas trouvé dans src/main/resources");
                return;
            }
            document = parser.parse(inputStream);
        } catch (Exception e) {
            System.err.println("Erreur lors du chargement du PDF : " + e.getMessage());
            return;
        }
        System.out.println("Document PDF chargé.");

        // 2. Découper le document en morceaux (chunks)
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 20);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("Document découpé en " + segments.size() + " segments.");

        // 3. Créer le modèle d'embedding (LOCAL)
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        System.out.println("Modèle d'embedding local chargé.");

        // 4. Créer les embeddings pour les segments
        System.out.println("Création des embeddings (vecteurs)...");
        Response<List<Embedding>> embeddingsResponse = embeddingModel.embedAll(segments);
        List<Embedding> embeddings = embeddingsResponse.content();
        System.out.println("Embeddings créés.");

        // 5. Stocker les embeddings en mémoire
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddings, segments);
        System.out.println("Embeddings stockés en mémoire.");


        // --- PHASE 2 : GÉNÉRATION (Conversation) ---
        System.out.println("Phase 2 : Préparation de l'assistant...");

        // 6. Récupérer la clé API (pour le ChatModel GEMINI)
        String geminiApiKey = System.getenv("GEMINI_KEY");
        if (geminiApiKey == null || geminiApiKey.isEmpty()) {
            System.err.println("Erreur : La variable d'environnement GEMINI_KEY n'est pas définie.");
            return;
        }

        // 7. Créer le ChatModel (Gemini)
        // Note : logRequests() et logResponses() ne sont pas disponibles en 0.35.0
        // Le logging est géré via configureLogger() ci-dessus
        ChatLanguageModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiApiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3) // Température basse pour le RAG
                .build();
        System.out.println("ChatModel Gemini chargé.");

        // 8. Créer le Content Retriever (le "chercheur" de RAG)
        ContentRetriever contentRetriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // 9. Créer l'Assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(contentRetriever)
                .build();

        System.out.println("Assistant RAG prêt. Posez vos questions sur le PDF (tapez 'fin' pour quitter).");

        // 10. Lancer la boucle de conversation
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