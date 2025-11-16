package ma.emsi.QejiouSalaheddine.tp4;

/**
 * Interface pour l'assistant IA, comme demandé dans le TP.
 * LangChain4j l'implémentera automatiquement.
 */
public interface Assistant {
    String chat(String userMessage);
}
