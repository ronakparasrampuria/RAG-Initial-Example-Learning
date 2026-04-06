package com.eazybytes.springai.rag;

import jakarta.annotation.PostConstruct;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.stereotype.Component;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

//@Component
public class RandomDataLoader {

    private static final Logger log = LoggerFactory.getLogger(RandomDataLoader.class);

    private final VectorStore vectorStore;
    private final EmbeddingModel embeddingModel;

    public RandomDataLoader(VectorStore vectorStore, EmbeddingModel embeddingModel) {
        this.vectorStore = vectorStore;
        this.embeddingModel = embeddingModel;
    }

    @PostConstruct
    public void loadSentencesIntoVectorStore() {
        List<String> sentences = List.of(
                "Java is used for building scalable enterprise applications.",
                "Python is commonly used for machine learning and automation tasks.",
                "JavaScript is essential for creating interactive web pages.",
                "Docker packages applications into lightweight containers.",
                "Kubernetes automates container orchestration at scale.",
                "Redis is an in-memory data store used for caching.",
                "PostgreSQL supports complex queries and full ACID compliance.",
                "Kafka is a distributed event streaming platform.",
                "REST APIs allow stateless client-server communication.",
                "GraphQL enables clients to fetch exactly the data they need.",
                "Credit scores influence the interest rates on loans.",
                "Mutual funds pool money from investors to buy securities.",
                "Bitcoin operates on a decentralized peer-to-peer network.",
                "Ethereum supports smart contract deployment.",
                "The stock market opens at 9:30 a.m. EST on weekdays.",
                "Compound interest increases investment returns over time.",
                "Diversifying investments reduces overall risk.",
                "A blockchain is a distributed, immutable ledger of transactions.",
                "Photosynthesis is how plants convert sunlight into energy.",
                "The water cycle involves evaporation, condensation, and precipitation.",
                "The ozone layer protects Earth from harmful ultraviolet rays.",
                "Earth revolves around the Sun in an elliptical orbit.",
                "Lightning is a discharge of electricity caused by charged clouds.",
                "DNA is the molecule that carries genetic instructions in living organisms.",
                "Volcanoes form when magma rises through Earth's crust.",
                "Earthquakes are caused by sudden tectonic shifts.",
                "The Sahara is the largest hot desert in the world.",
                "Mount Kilimanjaro is the tallest mountain in Africa.",
                "Japan is known for its cherry blossoms and advanced technology.",
                "The Great Wall of China is over 13,000 miles long.",
                "Niagara Falls is located between Canada and the U.S.",
                "The Amazon River is the second longest river in the world.",
                "Oats are high in fiber and help reduce cholesterol.",
                "Drinking water improves digestion and skin health.",
                "A balanced diet includes proteins, carbs, fats, and vitamins.",
                "Broccoli is rich in vitamins A, C, and K.",
                "Green tea contains antioxidants beneficial for metabolism.",
                "Too much sugar increases the risk of diabetes.",
                "Walking 30 minutes a day improves cardiovascular health.",
                "Meditation can reduce stress and improve focus.",
                "Gratitude journaling is linked to higher happiness levels.",
                "Deep breathing exercises help regulate anxiety.",
                "Reading daily improves vocabulary and cognitive function.",
                "Setting daily goals increases productivity.",
                "STEM stands for Science, Technology, Engineering, and Mathematics.",
                "Bloom’s taxonomy categorizes educational goals.",
                "Project-based learning enhances student engagement.",
                "Online courses offer flexibility for remote learners.",
                "Flashcards are effective for memorizing vocabulary.",
                "Agile methodology promotes iterative software development.",
                "OKRs help align team goals with business strategy.",
                "Remote work offers flexibility but requires clear communication.",
                "CRM systems manage customer relationships and sales pipelines.",
                "SWOT analysis identifies strengths, weaknesses, opportunities, and threats."
        );
        List<Document> documents = sentences.stream().map(Document::new).collect(Collectors.toList());

        // === Log embedding details for the first 3 sentences ===
        log.info("========== EMBEDDING DETAILS ==========");
        log.info("Embedding Model class: {}", embeddingModel.getClass().getSimpleName());
        log.info("Total documents to embed: {}", documents.size());

        // Generate embeddings manually for a few samples to inspect them
        List<String> sampleTexts = sentences.subList(0, Math.min(3, sentences.size()));
        EmbeddingResponse embeddingResponse = embeddingModel.embedForResponse(sampleTexts);

        embeddingResponse.getResults().forEach(embedding -> {
            float[] vector = embedding.getOutput();
            log.info("--------------------------------------------");
            log.info("Text index   : {}", embedding.getIndex());
            log.info("Text content : {}", sampleTexts.get(embedding.getIndex()));
            log.info("Vector dimensions : {}", vector.length);
            log.info("First 10 values   : {}", Arrays.toString(Arrays.copyOf(vector, Math.min(10, vector.length))));
            log.info("Last 5 values     : {}",
                    Arrays.toString(Arrays.copyOfRange(vector, Math.max(0, vector.length - 5), vector.length)));
        });

        log.info("========== METADATA FROM EMBEDDING RESPONSE ==========");
        if (embeddingResponse.getMetadata() != null) {
            log.info("Model used   : {}", embeddingResponse.getMetadata().getModel());
            log.info("Token usage  : {}", embeddingResponse.getMetadata().getUsage());
        }
        log.info("========== NOW STORING ALL {} DOCUMENTS IN VECTOR STORE ==========", documents.size());

        vectorStore.add(documents);

        log.info("========== ALL DOCUMENTS STORED SUCCESSFULLY ==========");
    }

}
