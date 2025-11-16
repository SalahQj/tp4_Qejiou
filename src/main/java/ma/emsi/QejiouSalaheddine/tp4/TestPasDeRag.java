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
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.io.InputStream;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Classe principale pour le Test 4 du TP 4 : RAG avec Routage personnalisé (Pas de RAG).
 */
public class TestPasDeRag {

    public static void main(String[] args) {

        // 1. Configurer le logging (pour voir la décision du routeur)
        configureLogger();

        // --- PHASE 1 : INGESTION (Identique au Test 1, mais une seule source) ---
        System.out.println("Phase 1 : Ingestion du document PDF (rag.pdf)...");

        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // Nous n'avons besoin que du retriever pour le RAG
        ContentRetriever ragRetriever = createRetriever("/rag.pdf", embeddingModel);
        System.out.println("Ingestion du rag.pdf terminée.");

        // --- PHASE 2 : CONFIGURATION DU ROUTEUR PERSONNALISÉ ---
        System.out.println("Phase 2 : Préparation de l'assistant avec routage personnalisé...");

        // 2. Créer le ChatModel (Gemini)
        String geminiApiKey = System.getenv("GEMINI_KEY");
        if (geminiApiKey == null || geminiApiKey.isEmpty()) {
            System.err.println("Erreur : GEMINI_KEY n'est pas définie.");
            return;
        }

        ChatLanguageModel chatModel = GoogleAiGeminiChatModel.builder()
                .apiKey(geminiApiKey)
                .modelName("gemini-2.5-flash")
                .temperature(0.0)
                .build();
        System.out.println("ChatModel Gemini chargé.");

        // 3. (BONUS) Créer le PromptTemplate pour le routeur - VERSION AMÉLIORÉE
        PromptTemplate promptTemplate = PromptTemplate.from(
                "Analyse cette question : '{{question}}'\n\n" +
                        "Cette question porte-t-elle SPÉCIFIQUEMENT et DIRECTEMENT sur l'un de ces sujets techniques :\n" +
                        "- RAG (Retrieval Augmented Generation)\n" +
                        "- LangChain4j\n" +
                        "- Fine-tuning de modèles d'IA\n" +
                        "- Architecture d'intelligence artificielle\n" +
                        "- Embeddings ou vecteurs\n\n" +
                        "Réponds UNIQUEMENT par 'oui' ou 'non'.\n" +
                        "Réponds 'non' pour :\n" +
                        "- Les salutations (bonjour, salut, etc.)\n" +
                        "- Les questions sur la cuisine, recettes, alimentation\n" +
                        "- Les questions générales non techniques\n" +
                        "- Toute question qui n'est pas directement liée aux sujets techniques listés ci-dessus\n\n" +
                        "Réponse :"
        );

        // 4. Créer le QueryRouter personnalisé (en utilisant une classe interne)
        class QueryRouterPourEviterRag implements QueryRouter {
            @Override
            public List<ContentRetriever> route(Query query) {
                // Créer le prompt en utilisant le template
                Prompt prompt = promptTemplate.apply(Map.of("question", query.text()));

                // Demander au LLM de classifier la question
                String reponse = chatModel.generate(prompt.text());

                System.out.println("Décision du Routeur : Question = '" + query.text() + "' -> Réponse IA = '" + reponse.trim() + "'");

                if (reponse.toLowerCase().trim().startsWith("non")) {
                    // La question est "Bonjour" ou hors sujet.
                    // On ne fait PAS de RAG.
                    System.out.println("Résultat Routage : PAS DE RAG");
                    return Collections.emptyList(); // Retourne une liste vide
                } else {
                    // La question concerne l'IA.
                    // On active le RAG.
                    System.out.println("Résultat Routage : ACTIVATION RAG (rag.pdf)");
                    return Collections.singletonList(ragRetriever); // Retourne le retriever du PDF
                }
            }
        }

        // 5. Créer le RetrievalAugmentor avec notre routeur personnalisé
        RetrievalAugmentor retrievalAugmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(new QueryRouterPourEviterRag())
                .build();

        // 6. Créer l'Assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatLanguageModel(chatModel)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .retrievalAugmentor(retrievalAugmentor)
                .build();

        System.out.println("Assistant RAG avec Routage Personnalisé prêt. (tapez 'fin' pour quitter).");

        // 7. Lancer la boucle de conversation
        conversationAvec(assistant);
    }


    // --- MÉTHODES UTILITAIRES (copiées des tests précédents) ---

    /**
     * Méthode utilitaire pour l'ingestion d'un document.
     */
    private static ContentRetriever createRetriever(String resourceName, EmbeddingModel embeddingModel) {
        ApacheTikaDocumentParser parser = new ApacheTikaDocumentParser();
        Document document;
        try (InputStream inputStream = TestPasDeRag.class.getResourceAsStream(resourceName)) {
            if (inputStream == null) {
                throw new RuntimeException("Erreur : Le fichier " + resourceName + " n'est pas trouvé.");
            }
            document = parser.parse(inputStream);
        } catch (Exception e) {
            throw new RuntimeException("Erreur lors du chargement de " + resourceName, e);
        }
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 20);
        List<TextSegment> segments = splitter.split(document);
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
        Response<List<Embedding>> embeddingsResponse = embeddingModel.embedAll(segments);
        embeddingStore.addAll(embeddingsResponse.content(), segments);
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