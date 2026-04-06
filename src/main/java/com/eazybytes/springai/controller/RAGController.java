package com.eazybytes.springai.controller;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.embedding.EmbeddingResponse;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.http.ResponseEntity;
import org.springframework.lang.Nullable;
import org.springframework.web.bind.annotation.*;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.springframework.ai.chat.memory.ChatMemory.CONVERSATION_ID;

@RestController
@RequestMapping("/api/rag")
public class RAGController {

    private static final Logger log = LoggerFactory.getLogger(RAGController.class);

    private final ChatClient chatClient;
    private final VectorStore vectorStore;
    private final EmbeddingModel embeddingModel;

    @Value("classpath:/promptTemplates/systemPromptRandomDataTemplate.st")
    Resource promptTemplate;

    @Value("classpath:/promptTemplates/systemPromptTemplate.st")
    Resource hrSystemTemplate;

    public RAGController(@Qualifier("chatMemoryChatClient") ChatClient chatClient,
                         VectorStore vectorStore, EmbeddingModel embeddingModel) {
        this.chatClient = chatClient;
        this.vectorStore = vectorStore;
        this.embeddingModel = embeddingModel;
    }

    @GetMapping("/random/chat")
    public ResponseEntity<String> randomChat(@RequestHeader("username") String username,
                                             @RequestParam("message") String message) {

        // === Log the query embedding ===
        log.info("========== QUERY EMBEDDING DETAILS ==========");
        log.info("User query: {}", message);
        EmbeddingResponse queryEmbeddingResponse = embeddingModel.embedForResponse(List.of(message));
        queryEmbeddingResponse.getResults().forEach(embedding -> {
            float[] vector = embedding.getOutput();
            log.info("Query vector dimensions : {}", vector.length);
            log.info("Query first 10 values   : {}", Arrays.toString(Arrays.copyOf(vector, Math.min(10, vector.length))));
        });
        if (queryEmbeddingResponse.getMetadata() != null) {
            log.info("Embedding model used    : {}", queryEmbeddingResponse.getMetadata().getModel());
            log.info("Embedding token usage   : {}", queryEmbeddingResponse.getMetadata().getUsage());
        }

        // === Perform similarity search ===
        SearchRequest searchRequest =
                SearchRequest.builder().query(message).topK(3).similarityThreshold(0.5).build();
        List<Document> similarDocs = vectorStore.similaritySearch(searchRequest);

        // === Log retrieved documents with similarity scores ===
        log.info("========== SIMILARITY SEARCH RESULTS ==========");
        log.info("Documents found: {}", similarDocs.size());
        for (int i = 0; i < similarDocs.size(); i++) {
            Document doc = similarDocs.get(i);
            log.info("--- Result #{} ---", i + 1);
            log.info("  Text     : {}", doc.getText());
            log.info("  Score    : {}", doc.getScore());
            log.info("  Metadata : {}", doc.getMetadata());
            log.info("  Doc ID   : {}", doc.getId());
        }
        log.info("================================================");

        String similarContext = similarDocs.stream()
                .map(Document::getText)
                .collect(Collectors.joining(System.lineSeparator()));
        String answer = chatClient.prompt()
                .system(promptSystemSpec -> promptSystemSpec.text(promptTemplate)
                        .param("documents", similarContext))
                .advisors(a -> a.param(CONVERSATION_ID, username))
                .user(message)
                .call().content();
        return ResponseEntity.ok(answer);
    }

    @GetMapping("/document/chat")
    public ResponseEntity<String> documentChat(@RequestHeader("username") String username,
                                               @RequestParam("message") String message) {
        SearchRequest searchRequest =
                SearchRequest.builder().query(message).topK(3).similarityThreshold(0.5).build();
        List<Document> similarDocs =  vectorStore.similaritySearch(searchRequest);
        String similarContext = similarDocs.stream()
                .map(Document::getText)
                .collect(Collectors.joining(System.lineSeparator()));
        String answer = chatClient.prompt()
                .system(promptSystemSpec -> promptSystemSpec.text(hrSystemTemplate)
                                .param("documents", similarContext))
                .advisors(a -> a.param(CONVERSATION_ID, username))
                .user(message)
                .call().content();
        return ResponseEntity.ok(answer);
    }

    // When using Retrieval Augmentation Advisors, the chat client will automatically perform the retrieval step based on the query and the provided configuration. In that case, you can have a simpler endpoint like this:
    @GetMapping("/document/augmented-advisor/chat")
    public ResponseEntity<String> documentChatUsingRetrievalAugmentedAdvisor(@RequestHeader("username") String username,
                                               @RequestParam("message") String message) {
        /*SearchRequest searchRequest =
                SearchRequest.builder().query(message).topK(3).similarityThreshold(0.5).build();
        List<Document> similarDocs =  vectorStore.similaritySearch(searchRequest);
        String similarContext = similarDocs.stream()
                .map(Document::getText)
                .collect(Collectors.joining(System.lineSeparator()));*/
        String answer = chatClient.prompt()
                /*.system(promptSystemSpec -> promptSystemSpec.text(hrSystemTemplate)
                        .param("documents", similarContext))*/
                .advisors(a -> a.param(CONVERSATION_ID, username))
                .user(message)
                .call().content();
        return ResponseEntity.ok(answer);
    }
}
