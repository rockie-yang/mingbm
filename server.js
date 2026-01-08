import { createServer } from 'http';
import { readFileSync, existsSync } from 'fs';
import { WebSocketServer } from 'ws';
import { extname, join } from 'path';

const PORT = 3000;
const ROOT = '.';

const mimeTypes = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.svg': 'image/svg+xml'
};

// HTTP server for static files
const server = createServer((req, res) => {
    let filePath = join(ROOT, req.url === '/' ? 'index.html' : req.url.split('?')[0]);

    if (!existsSync(filePath)) {
        res.writeHead(404);
        res.end('Not found');
        return;
    }

    const ext = extname(filePath);
    const contentType = mimeTypes[ext] || 'application/octet-stream';

    res.writeHead(200, { 'Content-Type': contentType });
    res.end(readFileSync(filePath));
});

// WebSocket server for slide sync
const wss = new WebSocketServer({ server });
const clients = new Set();

wss.on('connection', (ws) => {
    clients.add(ws);
    console.log('Client connected, total:', clients.size);

    ws.on('message', (data) => {
        const msg = JSON.parse(data);
        // Broadcast to all other clients
        clients.forEach((client) => {
            if (client !== ws && client.readyState === 1) {
                client.send(JSON.stringify(msg));
            }
        });
    });

    ws.on('close', () => {
        clients.delete(ws);
        console.log('Client disconnected, total:', clients.size);
    });
});

server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}`);
    console.log('Open multiple tabs to sync slides via WebSocket');
});
