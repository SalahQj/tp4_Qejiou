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
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.WebSearchEngine;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;

import java.io.InputStream;
import java.time.Duration;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Classe principale pour le Test 5 du TP 4 : RAG avec recherche sur le Web (Tavily).
 */
public class TestWebSearch {

    public static void main(String[] args) {

        // 1. Configurer le logging
        configureLogger();

        System.out.println("Début du Test 5 : RAG avec recherche Web (Tavily)...");

        // --- PHASE 1 : INGESTION du document local (PDF) ---
        System.out.println("Phase 1 : Ingestion du document PDF (rag.pdf)...");

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();
        System.out.println("Modèle d'embedding local chargé.");

        // Créer le ContentRetriever pour le PDF local (avec moins de résultats)
        ContentRetriever pdfRetriever = createPdfRetriever("/rag.pdf", embeddingModel);
        System.out.println("ContentRetriever PDF créé.");

        // --- PHASE 2 : CONFIGURATION de la recherche Web ---
        System.out.println("Phase 2 : Configuration de la recherche Web...");

        // 2. Récupérer la clé API Tavily
        String tavilyApiKey = System.getenv("TAVILY_API_KEY");
        if (tavilyApiKey == null || tavilyApiKey.isEmpty()) {
            System.err.println("Erreur : La variable d'environnement TAVILY_API_KEY n'est pas définie.");
            System.err.println("Définissez-la avec : $env:TAVILY_API_KEY=\"votre_clé\"");
            return;
        }

        // 3. Créer le WebSearchEngine (Tavily) - sans maxResults
        WebSearchEngine webSearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(tavilyApiKey)
                .build();
        System.out.println("WebSearchEngine Tavily créé.");

        // 4. Créer le ContentRetriever pour le Web avec maxResults limité
        ContentRetriever webRetriever = WebSearchContentRetriever.builder()
                .webSearchEngine(webSearchEngine)
                .maxResults(2) // Limite à 2 résultats Web
                .build();
        System.out.println("ContentRetriever Web créé.");

        // --- PHASE 3 : CONFIGURATION du Routeur et de l'Assistant ---
        System.out.println("Phase 3 : Préparation de l'assistant...");

        // 5. Récupérer la clé API Gemini
        String geminiApiKey = System.getenv("GEMINI_KEY");
        if (geminiApiKey == null || geminiApiKey.isEmpty()) {
            System.err.println("Erreur : GEMINI_KEY n'est pas définie.");
            return;
        }

        // 6. Créer le ChatModel (Gemini) avec timeout plus long
        ChatLanguageModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiApiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .timeout(Duration.ofSeconds(120)) // Timeout de 120 secondes
                .build();
        System.out.println("ChatModel Gemini chargé.");

        // 7. Créer le QueryRouter qui utilise les 2 retrievers
        QueryRouter queryRouter = new DefaultQueryRouter(pdfRetriever, webRetriever);
        System.out.println("QueryRouter créé (PDF + Web).");

        // 8. Créer le RetrievalAugmentor
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(queryRouter)
                .build();

        // 9. Créer l'Assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        System.out.println("Assistant RAG avec recherche Web prêt. (tapez 'fin' pour quitter).");
        System.out.println("Les informations seront récupérées à la fois du PDF local ET du Web !");
        System.out.println("Note : Le premier appel peut prendre du temps (recherche Web + PDF).");

        // 10. Lancer la boucle de conversation
        conversationAvec(assistant);
    }

    /**
     * Méthode utilitaire pour créer un ContentRetriever à partir d'un PDF.
     */
    private static ContentRetriever createPdfRetriever(String resourceName, EmbeddingModel embeddingModel) {
        ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();
        Document document;
        try (InputStream inputStream = TestWebSearch.class.getResourceAsStream(resourceName)) {
            if (inputStream == null) {
                throw new RuntimeException("Erreur : Le fichier " + resourceName + " n'est pas trouvé.");
            }
            document = parser.parse(inputStream);
        } catch (Exception e) {
            throw new RuntimeException("Erreur lors du chargement de " + resourceName, e);
        }

        DocumentSplitter splitter = DocumentSplitters.recursive(300, 20);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("Document " + resourceName + " découpé en " + segments.size() + " segments.");

        Response<List<Embedding>> embeddingsResponse = embeddingModel.embedAll(segments);
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        embeddingStore.addAll(embeddingsResponse.content(), segments);

        return EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(1) // Réduire à 1 résultat du PDF pour éviter trop de données
                .minScore(0.6) // Score minimum plus élevé
                .build();
    }

    /**
     * Méthode pour configurer le logging
     */
    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
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
                System.out.println("Recherche en cours (PDF + Web)...");
                try {
                    String reponse = assistant.chat(question);
                    System.out.println("Assistant : " + reponse);
                } catch (Exception e) {
                    System.err.println("Erreur lors de la génération de la réponse : " + e.getMessage());
                    System.err.println("Veuillez réessayer avec une question plus courte ou différente.");
                }
            }
        }
    }
}