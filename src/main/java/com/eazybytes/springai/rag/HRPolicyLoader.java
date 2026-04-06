package com.eazybytes.springai.rag;

import jakarta.annotation.PostConstruct;
import org.apache.tika.Tika;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.transformer.splitter.TextSplitter;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Component;

import java.util.List;

@Component
public class HRPolicyLoader {

    private final VectorStore vectorStore;

    @Value("classpath:Eazybytes_HR_Policies.pdf")
    Resource policyFile;

    public HRPolicyLoader(VectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    @PostConstruct
    public void loadPDF() {
        TikaDocumentReader tikaDocumentReader = new TikaDocumentReader(policyFile);
        List<Document> docs = tikaDocumentReader.get();
        // This 200 and 400 are tokens
        // We are saying that we want to split the document into chunks of 200 tokens and we want to have a maximum of 400 chunks. This is to ensure that we don't have too many chunks and that we don't have chunks that are too small.
        TextSplitter textSplitter =
                TokenTextSplitter.builder().withChunkSize(200).withMaxNumChunks(400).build();
        vectorStore.add(textSplitter.split(docs));
    }
}
