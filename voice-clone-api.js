const http = require('http');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');
const url = require('url');
const querystring = require('querystring');

// 配置
const CONFIG = {
    port: 3000,
    uploadDir: './uploads',
    modelDir: './voice-models',
    outputDir: './outputs',
    sampleRate: 16000,
    frameSize: 1024,
    hopSize: 512,
    tempDir: './temp'
};

// 确保目录存在
function ensureDir(dir) {
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
}

// 初始化目录
function initDirectories() {
    ensureDir(CONFIG.uploadDir);
    ensureDir(CONFIG.modelDir);
    ensureDir(CONFIG.outputDir);
    ensureDir(CONFIG.tempDir);
}

// 检查系统工具
function checkSystemTools() {
    try {
        execSync('ffmpeg -version', { stdio: 'ignore' });
        return true;
    } catch (e) {
        console.error('错误: 请安装ffmpeg以继续');
        console.error('安装指南: https://ffmpeg.org/download.html');
        process.exit(1);
    }
}

// 解析multipart/form-data
function parseMultipart(buffer, boundary) {
    const parts = buffer.toString().split(`--${boundary}`);
    const result = {};
    
    for (const part of parts) {
        if (part.trim() === '' || part.includes('--')) continue;
        
        const [headersPart, ...bodyParts] = part.split('\r\n\r\n');
        const body = bodyParts.join('\r\n\r\n').replace(/\r\n$/, '');
        
        const disposition = headersPart.match(/Content-Disposition: form-data; name="([^"]+)"(?:; filename="([^"]+)")?/);
        if (disposition) {
            const [, name, filename] = disposition;
            
            if (filename) {
                // 处理文件
                const contentTypeMatch = headersPart.match(/Content-Type: ([^\r]+)/);
                const contentType = contentTypeMatch ? contentTypeMatch[1] : 'application/octet-stream';
                
                // 提取二进制数据
                const bodyBuffer = Buffer.from(body, 'binary');
                
                result[name] = {
                    filename,
                    contentType,
                    data: bodyBuffer
                };
            } else {
                // 处理普通字段
                result[name] = body;
            }
        }
    }
    
    return result;
}

// 转换音频格式
function convertAudio(inputPath, outputPath) {
    execSync(
        `ffmpeg -y -i "${inputPath}" -ar ${CONFIG.sampleRate} -ac 1 -f wav "${outputPath}"`,
        { stdio: 'ignore' }
    );
}

// 读取WAV文件数据
function readWavData(filePath) {
    const buffer = fs.readFileSync(filePath);
    
    // 验证WAV文件头
    if (buffer.toString('ascii', 0, 4) !== 'RIFF' || 
        buffer.toString('ascii', 8, 12) !== 'WAVE') {
        throw new Error('无效的WAV文件');
    }
    
    return {
        header: buffer.slice(0, 44),
        data: new Int16Array(buffer.slice(44).buffer)
    };
}

// 保存WAV数据
function saveWavData(filePath, header, data) {
    const buffer = Buffer.alloc(44 + data.byteLength);
    header.copy(buffer, 0, 0, 44);
    Buffer.from(data.buffer).copy(buffer, 44);
    fs.writeFileSync(filePath, buffer);
}

// 提取音频特征
function extractFeatures(audioData) {
    const features = [];
    const numFrames = Math.floor((audioData.length - CONFIG.frameSize) / CONFIG.hopSize) + 1;
    
    for (let i = 0; i < numFrames; i++) {
        const start = i * CONFIG.hopSize;
        const end = Math.min(start + CONFIG.frameSize, audioData.length);
        const frame = audioData.subarray(start, end);
        
        features.push([
            calculateEnergy(frame),
            estimatePitch(frame),
            estimateSpectralCentroid(frame)
        ]);
    }
    
    return features;
}

// 计算能量
function calculateEnergy(frame) {
    let sum = 0;
    for (const sample of frame) sum += sample * sample;
    return sum / frame.length;
}

// 估计音高
function estimatePitch(frame) {
    let zeroCrossings = 0;
    for (let i = 1; i < frame.length; i++) {
        if ((frame[i-1] >= 0 && frame[i] < 0) || (frame[i-1] < 0 && frame[i] >= 0)) {
            zeroCrossings++;
        }
    }
    return zeroCrossings;
}

// 估计频谱质心
function estimateSpectralCentroid(frame) {
    let weightedSum = 0, sum = 0;
    for (let i = 0; i < frame.length; i++) {
        const val = Math.abs(frame[i]);
        sum += val;
        weightedSum += i * val;
    }
    return sum > 0 ? weightedSum / sum : 0;
}

// 计算平均特征
function calculateAverageFeatures(features) {
    const sums = [0, 0, 0];
    const counts = [0, 0, 0];
    
    for (const feature of features) {
        for (let i = 0; i < 3; i++) {
            if (!isNaN(feature[i])) {
                sums[i] += feature[i];
                counts[i]++;
            }
        }
    }
    
    return [
        counts[0] > 0 ? sums[0] / counts[0] : 0,
        counts[1] > 0 ? sums[1] / counts[1] : 0,
        counts[2] > 0 ? sums[2] / counts[2] : 0
    ];
}

// 保存模型
function saveModel(userId, features, header) {
    const userDir = path.join(CONFIG.modelDir, userId);
    ensureDir(userDir);
    
    fs.writeFileSync(
        path.join(userDir, 'features.json'),
        JSON.stringify(features)
    );
    fs.writeFileSync(
        path.join(userDir, 'header.bin'),
        header
    );
    
    const avgFeatures = calculateAverageFeatures(features);
    fs.writeFileSync(
        path.join(userDir, 'signature.json'),
        JSON.stringify(avgFeatures)
    );
    
    return userDir;
}

// 加载模型
function loadModel(userId) {
    const userDir = path.join(CONFIG.modelDir, userId);
    if (!fs.existsSync(userDir)) {
        throw new Error(`用户 ${userId} 的模型不存在`);
    }
    
    return {
        features: JSON.parse(fs.readFileSync(path.join(userDir, 'features.json'))),
        header: fs.readFileSync(path.join(userDir, 'header.bin')),
        signature: JSON.parse(fs.readFileSync(path.join(userDir, 'signature.json')))
    };
}

// 应用声音转换
function applyVoiceTransformation(baseAudio, modelSignature) {
    const transformed = new Int16Array(baseAudio.length);
    const [targetEnergy, targetPitch, targetCentroid] = modelSignature;
    
    // 计算基础音频特征
    const baseFeatures = extractFeatures(baseAudio);
    const [baseEnergy, basePitch, baseCentroid] = calculateAverageFeatures(baseFeatures);
    
    // 计算转换因子
    const energyFactor = baseEnergy > 0 ? Math.min(3, Math.max(0.3, targetEnergy / baseEnergy)) : 1;
    const pitchFactor = basePitch > 0 ? Math.min(2, Math.max(0.5, targetPitch / basePitch)) : 1;
    const centroidFactor = baseCentroid > 0 ? Math.min(2, Math.max(0.5, targetCentroid / baseCentroid)) : 1;
    
    // 应用转换
    for (let i = 0; i < baseAudio.length; i++) {
        // 简单的音调转换 (基于位置的偏移)
        const pos = Math.round(i * pitchFactor);
        const sample = pos < baseAudio.length ? baseAudio[pos] : 0;
        
        // 应用能量调整
        transformed[i] = Math.max(-32768, Math.min(32767, sample * energyFactor));
    }
    
    return transformed;
}

// 生成基础语音
function generateBaseVoice(text, outputPath) {
    try {
        // Windows
        execSync(`powershell -Command "Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).SetOutputToWaveFile('${outputPath}'); (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('${text.replace(/'/g, "''")}')"`, { stdio: 'ignore' });
    } catch (e) {
        try {
            // macOS
            execSync(`say -o "${outputPath}" --data-format=LEI16@16000 "${text}"`, { stdio: 'ignore' });
        } catch (e) {
            // Linux
            execSync(`espeak -v en-us -s 150 -w "${outputPath}" "${text}"`, { stdio: 'ignore' });
        }
    }
}

// 处理训练请求
async function handleTrainRequest(userId, audioBuffer, response) {
    try {
        // 保存上传的音频
        const inputPath = path.join(CONFIG.uploadDir, `${userId}-train-input.wav`);
        fs.writeFileSync(inputPath, audioBuffer);
        
        // 转换为标准格式
        const convertedPath = path.join(CONFIG.tempDir, `${userId}-train-converted.wav`);
        convertAudio(inputPath, convertedPath);
        
        // 读取音频数据
        const { header, data } = readWavData(convertedPath);
        
        // 提取特征并保存模型
        const features = extractFeatures(data);
        const modelPath = saveModel(userId, features, header);
        
        response.writeHead(200, { 'Content-Type': 'application/json' });
        response.end(JSON.stringify({
            success: true,
            message: '模型训练完成',
            userId,
            modelPath
        }));
    } catch (error) {
        response.writeHead(500, { 'Content-Type': 'application/json' });
        response.end(JSON.stringify({
            success: false,
            error: error.message
        }));
    }
}

// 处理克隆请求
async function handleCloneRequest(userId, text, response) {
    try {
        // 加载模型
        const model = loadModel(userId);
        
        // 生成基础语音
        const baseVoicePath = path.join(CONFIG.tempDir, `${userId}-base-voice.wav`);
        generateBaseVoice(text, baseVoicePath);
        
        // 转换为标准格式
        const convertedPath = path.join(CONFIG.tempDir, `${userId}-converted-voice.wav`);
        convertAudio(baseVoicePath, convertedPath);
        
        // 读取并转换音频
        const { data: baseAudio } = readWavData(convertedPath);
        const transformedAudio = applyVoiceTransformation(baseAudio, model.signature);
        
        // 保存结果
        const outputPath = path.join(CONFIG.outputDir, `${userId}-${Date.now()}-cloned.wav`);
        saveWavData(outputPath, model.header, transformedAudio);
        
        // 读取结果文件并返回
        const outputData = fs.readFileSync(outputPath);
        
        response.writeHead(200, {
            'Content-Type': 'audio/wav',
            'Content-Disposition': `attachment; filename="cloned-voice.wav"`,
            'Content-Length': outputData.length
        });
        response.end(outputData);
    } catch (error) {
        response.writeHead(500, { 'Content-Type': 'application/json' });
        response.end(JSON.stringify({
            success: false,
            error: error.message
        }));
    }
}

// 创建服务器
function createServer() {
    return http.createServer((req, res) => {
        const parsedUrl = url.parse(req.url);
        const pathname = parsedUrl.pathname;
        
        // 处理CORS
        res.setHeader('Access-Control-Allow-Origin', '*');
        res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
        res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
        
        if (req.method === 'OPTIONS') {
            res.writeHead(204);
            res.end();
            return;
        }
        
        // 健康检查端点
        if (pathname === '/health' && req.method === 'GET') {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ status: 'ok', timestamp: new Date().toISOString() }));
            return;
        }
        
        // 训练模型端点
        if (pathname === '/train' && req.method === 'POST') {
            const query = querystring.parse(parsedUrl.query);
            const userId = query.userId;
            
            if (!userId) {
                res.writeHead(400, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ success: false, error: '缺少userId参数' }));
                return;
            }
            
            // 读取请求体
            const body = [];
            req.on('data', chunk => body.push(chunk));
            req.on('end', () => {
                const buffer = Buffer.concat(body);
                const contentType = req.headers['content-type'] || '';
                
                if (contentType.startsWith('multipart/form-data')) {
                    const boundary = contentType.split('boundary=')[1];
                    const parts = parseMultipart(buffer, boundary);
                    
                    if (parts.audio && parts.audio.data) {
                        handleTrainRequest(userId, parts.audio.data, res);
                    } else {
                        res.writeHead(400, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ success: false, error: '未找到音频数据' }));
                    }
                } else {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ success: false, error: '不支持的内容类型' }));
                }
            });
            return;
        }
        
        // 声音克隆端点
        if (pathname === '/clone' && req.method === 'POST') {
            let body = '';
            req.on('data', chunk => body += chunk);
            req.on('end', () => {
                try {
                    const data = JSON.parse(body);
                    
                    if (!data.userId || !data.text) {
                        res.writeHead(400, { 'Content-Type': 'application/json' });
                        res.end(JSON.stringify({ success: false, error: '缺少userId或text参数' }));
                        return;
                    }
                    
                    handleCloneRequest(data.userId, data.text, res);
                } catch (error) {
                    res.writeHead(400, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ success: false, error: '无效的JSON' }));
                }
            });
            return;
        }
        
        // 未找到端点
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ success: false, error: '端点不存在' }));
    });
}

// 启动服务
function startServer() {
    initDirectories();
    checkSystemTools();
    
    const server = createServer();
    server.listen(CONFIG.port, () => {
        console.log(`声音克隆API服务已启动，端口: ${CONFIG.port}`);
        console.log('可用端点:');
        console.log(`  POST http://localhost:${CONFIG.port}/train?userId=用户ID - 训练声音模型`);
        console.log(`  POST http://localhost:${CONFIG.port}/clone - 克隆声音`);
        console.log(`  GET http://localhost:${CONFIG.port}/health - 健康检查`);
    });
}

// 启动服务
startServer();
    