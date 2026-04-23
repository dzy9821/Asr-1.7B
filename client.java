package org.example;
import org.java_websocket.client.WebSocketClient;
import org.java_websocket.handshake.ServerHandshake;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Base64;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.*;

/**
 * 并发科大讯飞AST客户端 - 支持多音频文件并发识别
 */
public class ConcurrentAudioStreamingClient {

    // 配置参数
//

    private static final String WS_URL = "ws://localhost:8009/tuling/asr/v3";

//    private static final String WS_URL = "ws://148.148.52.26:10095";
    private static final String[] AUDIO_PATHS = {
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav",
        "120报警电话16k.wav"
            // 添加更多音频文件路径
//        "ttsmaker-file-2025-12-25-20-14-49.wav"
//            "iat.wav"
    };
    private static final int FRAME_SIZE = 4096;
    private static final long INTERVAL_MS = 40;
    private static final String OUTPUT_DIR = "concurrent_results";

    // 并发配置
    private static final int MAX_CONCURRENT_CLIENTS = 50;
    private static final int THREAD_POOL_SIZE = 50;

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final ExecutorService executorService = Executors.newFixedThreadPool(THREAD_POOL_SIZE);
    private final CountDownLatch allTasksLatch;

    public ConcurrentAudioStreamingClient() {
        this.allTasksLatch = new CountDownLatch(AUDIO_PATHS.length);
    }

    public static void main(String[] args) {
        ConcurrentAudioStreamingClient client = new ConcurrentAudioStreamingClient();
        try {
            client.startConcurrentRecognition();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void startConcurrentRecognition() throws Exception {
        System.out.println("开始并发语音识别，总任务数: " + AUDIO_PATHS.length);
        System.out.println("最大并发数: " + MAX_CONCURRENT_CLIENTS);

        long startTime = System.currentTimeMillis();

        // 使用信号量控制并发数
        Semaphore semaphore = new Semaphore(MAX_CONCURRENT_CLIENTS);

        // 提交所有任务
        List<Future<?>> futures = new ArrayList<>();
        for (int i = 0; i < AUDIO_PATHS.length; i++) {
            final int taskId = i;
            final String audioPath = AUDIO_PATHS[taskId];

            semaphore.acquire(); // 获取许可

            Future<?> future = executorService.submit(() -> {
                try {
                    processSingleAudio(taskId, audioPath);
                } catch (Exception e) {
                    System.err.println("任务 " + taskId + " 处理失败: " + e.getMessage());
                    e.printStackTrace();
                } finally {
                    semaphore.release(); // 释放许可
                    allTasksLatch.countDown();
                }
            });

            futures.add(future);
        }

        // 等待所有任务完成
        allTasksLatch.await();

        long totalTime = System.currentTimeMillis() - startTime;
        System.out.println("所有任务完成! 总耗时: " + totalTime + "ms");

        // 关闭线程池
        executorService.shutdown();
        executorService.awaitTermination(10, TimeUnit.SECONDS);

        System.out.println("并发客户端已关闭");
    }

    private void processSingleAudio(int taskId, String audioPath) throws Exception {
        System.out.println("任务 " + taskId + " 开始处理: " + audioPath);

        SingleAudioClient client = new SingleAudioClient(taskId, audioPath);
        client.process();
    }

    /**
     * 单个音频文件处理客户端
     */
    private class SingleAudioClient {
        private final int taskId;
        private final String audioPath;
        private final String outputFile;
        private final CountDownLatch completionLatch = new CountDownLatch(1);
        private long startTime = 0;

        public SingleAudioClient(int taskId, String audioPath) {
            this.taskId = taskId;
            this.audioPath = audioPath;
            this.outputFile = OUTPUT_DIR + "/task_" + taskId + "_results.json";

            // 创建输出目录
            new File(OUTPUT_DIR).mkdirs();
        }

        public void process() throws Exception {
            WebSocketClient client = createWebSocketClient();
            client.connect();

            // 等待连接建立
            Thread.sleep(1000);

            // 在单独的线程中发送数据
            Thread sendThread = new Thread(() -> {
                try {
                    sendAudioData(client);
                } catch (Exception e) {
                    System.err.println("任务 " + taskId + " 发送数据失败: " + e.getMessage());
                }
            });

            sendThread.start();

            // 等待识别完成
            completionLatch.await();

            // 延迟一小段时间再关闭连接，确保服务端能完成最后一条消息的发送
            // 避免出现 "bad Connection" 错误
            Thread.sleep(200);

            client.close();
            System.out.println("任务 " + taskId + " 处理完成");
        }

        private WebSocketClient createWebSocketClient() throws URISyntaxException {
            return new WebSocketClient(new URI(WS_URL)) {
                private int sequenceNumber = 1;
                private StringBuilder accumulatedText = new StringBuilder();
                private List<Object> results = new ArrayList<>();
                private String role = "(角色未知)";

                @Override
                public void onOpen(ServerHandshake handshake) {
                    System.out.println("任务 " + taskId + " WebSocket连接已建立");
                }

                @Override
                public void onMessage(String message) {
                    System.out.println(message);
                    try {
                        long currentTime = System.currentTimeMillis();
                        long elapsedTime = startTime > 0 ? currentTime - startTime : 0;

                        JsonNode resp = objectMapper.readTree(message);

                        // 添加序号到结果中
                        ObjectNode resultWithSequence = objectMapper.createObjectNode();
                        resultWithSequence.put("taskId", taskId);
                        resultWithSequence.put("sequence", sequenceNumber);
                        resultWithSequence.put("timestamp", System.currentTimeMillis() / 1000.0);
                        resultWithSequence.put("elapsed_ms", elapsedTime);
                        resultWithSequence.set("data", resp);

                        results.add(resultWithSequence);

                        // 解析识别结果
                        if (resp.has("payload") && resp.get("payload").has("result")) {
                            JsonNode result = resp.get("payload").get("result");
                            String msgtype = result.has("msgtype") ? result.get("msgtype").asText() : "";

                            // 提取当前帧的文本内容
                            StringBuilder currentText = new StringBuilder();
                            if (result.has("ws")) {
                                for (JsonNode wsItem : result.get("ws")) {
                                    if (wsItem.has("cw")) {
                                        for (JsonNode cwItem : wsItem.get("cw")) {
                                            if (cwItem.has("w")) {
                                                String rl = cwItem.has("rl") ? cwItem.get("rl").asText() : "";
                                                if (!rl.equals("0")) {
                                                    role = "(角色" + rl + ")";
                                                }
                                                currentText.append(cwItem.get("w").asText()).append(role);
                                            }
                                        }
                                    }
                                }
                            }

                            // 根据消息类型处理累积文本
                            String statusLabel;
                            String displayText;

                            if ("progressive".equals(msgtype)) {
                                statusLabel = "【中间状态】";
                                displayText = accumulatedText.toString() + currentText.toString();
                            } else if ("sentence".equals(msgtype)) {
                                accumulatedText.append(currentText.toString());
                                statusLabel = "【最终状态】";
                                displayText = accumulatedText.toString();
                            } else {
                                statusLabel = "【未知状态】";
                                displayText = accumulatedText.toString() + currentText.toString();
                            }

                            // 打印累积的文本内容
                            if (currentText.length() > 0 || accumulatedText.length() > 0) {
                                System.out.printf("任务%d 收到结果#%d %s [耗时: %dms]: %s%n",
                                        taskId, sequenceNumber, statusLabel, elapsedTime, displayText);
                            }
                        }

                        // 判断是否结束
                        if (resp.has("header") &&
                                resp.get("header").has("status") &&
                                resp.get("header").get("status").asInt() == 2) {
                            System.out.println("任务 " + taskId + " 识别结束，总耗时: " + elapsedTime + "ms");

                            // 保存结果到JSON文件
                            try (FileOutputStream fos = new FileOutputStream(outputFile)) {
                                objectMapper.writerWithDefaultPrettyPrinter()
                                        .writeValue(fos, results);
                                System.out.println("任务 " + taskId + " 结果已保存: " + outputFile);
                            } catch (IOException e) {
                                System.err.println("任务 " + taskId + " 保存结果文件失败: " + e.getMessage());
                            }

                            System.out.println("任务 " + taskId + " 最终累积文本: " + accumulatedText.toString());
                            
                            // 延迟一小段时间再触发关闭，确保服务端能完成消息发送
                            // 避免出现 "bad Connection" 错误
                            new Thread(() -> {
                                try {
                                    Thread.sleep(100);
                                    completionLatch.countDown();
                                } catch (InterruptedException e) {
                                    Thread.currentThread().interrupt();
                                    completionLatch.countDown();
                                }
                            }).start();
                        }

                        sequenceNumber++;

                    } catch (Exception e) {
                        System.err.println("任务 " + taskId + " 解析服务端消息失败: " + e.getMessage());
                    }
                }

                @Override
                public void onClose(int code, String reason, boolean remote) {
                    System.out.printf("任务%d WebSocket连接关闭: code=%d, reason=%s%n",
                            taskId, code, reason);
                    completionLatch.countDown();
                }

                @Override
                public void onError(Exception ex) {
                    System.err.println("任务 " + taskId + " WebSocket错误: " + ex.getMessage());
                    completionLatch.countDown();
                }
            };
        }

        private void sendAudioData(WebSocketClient client) {
            String traceId = UUID.randomUUID().toString();
            String bizId = "task_" + taskId + "_bizid";
            String appId = "123456";
            int status = 0;
            boolean firstMessageSent = false;

            File audioFile = new File(audioPath);
            if (!audioFile.exists()) {
                System.err.println("任务 " + taskId + " 音频文件不存在: " + audioPath);
                completionLatch.countDown();
                return;
            }

            try (FileInputStream fis = new FileInputStream(audioFile)) {
                while (true) {
                    byte[] data = new byte[FRAME_SIZE];
                    int bytesRead = fis.read(data);
                    if (bytesRead == -1) {
                        break;
                    }

                    // 检查是否为最后一块数据
                    long currentPos = fis.getChannel().position();
                    long fileSize = audioFile.length();
                    boolean isLastChunk = currentPos + bytesRead >= fileSize;

                    if (isLastChunk) {
                        status = 2;
                    }

                    // 处理实际读取的数据
                    byte[] actualData = new byte[bytesRead];
                    System.arraycopy(data, 0, actualData, 0, bytesRead);
                    String audioB64 = Base64.getEncoder().encodeToString(actualData);

                    // 构建消息
                    ObjectNode payload = objectMapper.createObjectNode();
                    ObjectNode audioNode = objectMapper.createObjectNode();
                    audioNode.put("audio", audioB64);
                    payload.set("audio", audioNode);

                    ObjectNode msg = objectMapper.createObjectNode();

                    // header
                    ObjectNode header = objectMapper.createObjectNode();
                    header.put("traceId", traceId);
                    header.put("appId", appId);
                    header.put("bizId", bizId);
                    header.put("status", status);
                    header.set("resIdList", objectMapper.createArrayNode());
                    msg.set("header", header);

                    // parameter
                    ObjectNode parameter = objectMapper.createObjectNode();
                    ObjectNode engine = objectMapper.createObjectNode();
                    engine.put("wdec_param_LanguageTypeChoice", "5");
                    parameter.set("engine", engine);
                    msg.set("parameter", parameter);

                    // payload
                    msg.set("payload", payload);

                    // 发送消息
                    String message = msg.toString();

                    // 记录第一次发送的时间
                    if (!firstMessageSent) {
                        startTime = System.currentTimeMillis();
                        System.out.println("任务 " + taskId + " 开始发送音频数据，时间标记: 0ms");
                        firstMessageSent = true;
                    }

                    client.send(message);

                    // 如果是最后一块数据，发送后立即退出循环
                    if (isLastChunk || status == 2) {
                        break;
                    }

                    // 等待指定间隔
                    try {
                        Thread.sleep(INTERVAL_MS);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    }

                    status = 1;
                }

            } catch (IOException e) {
                System.err.println("任务 " + taskId + " 读取音频文件失败: " + e.getMessage());
                completionLatch.countDown();
            }
        }
    }
}