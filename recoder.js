const fs = require('fs');
const readline = require('readline');
const dgram = require('dgram');

// List of public STUN servers (without protocol prefix in the array)
const STUN_SERVERS = [
    { host: 'stun.l.google.com', port: 19302 },
    { host: 'stun1.l.google.com', port: 19302 },
    { host: 'stun2.l.google.com', port: 19302 },
    { host: 'stun3.l.google.com', port: 19302 },
    { host: 'stun4.l.google.com', port: 19302 },
    { host: 'stun.relay.metered.ca', port: 80 },
    { host: 'stun.cloudflare.com', port: 3478 }
];

// File to store the results
const RESULTS_FILE = 'ip_records.json';

// Add new constant for ports file
const PORTS_FILE = 'port_records.json';

// Load existing records or create new array
function loadRecords() {
    try {
        const data = fs.readFileSync(RESULTS_FILE);
        return JSON.parse(data);
    } catch (error) {
        return [];
    }
}

// Save records to file
function saveRecords(records) {
    fs.writeFileSync(RESULTS_FILE, JSON.stringify(records, null, 2));
}

// Add function to load ports
function loadPorts() {
    try {
        const data = fs.readFileSync(PORTS_FILE);
        return JSON.parse(data);
    } catch (error) {
        return [];
    }
}

// Add function to save ports
function savePorts(records) {
    // Extract only the ports and timestamps
    const portRecords = records.map(record => ({
        port: record.port,
        timestamp: record.timestamp
    }));
    fs.writeFileSync(PORTS_FILE, JSON.stringify(portRecords, null, 2));
}

// Get public IP and port using STUN
async function getPublicIP(serverIndex) {
    const server = STUN_SERVERS[serverIndex];
    
    return new Promise((resolve, reject) => {
        const socket = dgram.createSocket('udp4');
        
        // Create STUN message (same as in p2p_transfer.js)
        const stunMessage = Buffer.from([
            0x00, 0x01, // Binding Request
            0x00, 0x00, // Message Length
            0x21, 0x12, 0xA4, 0x42, // Magic Cookie
            ...Array(12).fill(0) // Transaction ID
        ]);

        const timeout = setTimeout(() => {
            socket.close();
            reject(new Error('STUN request timed out'));
        }, 3000);

        socket.on('message', (msg, rinfo) => {
            clearTimeout(timeout);
            
            if (msg[0] === 0x01 && msg[1] === 0x01) {
                const port = msg.readUInt16BE(26) ^ 0x2112;
                const ip = Array.from(msg.slice(28, 32))
                    .map(b => b ^ 0x21)
                    .join('.');
                
                socket.close();
                resolve({
                    ip,
                    port,
                    server: `stun:${server.host}:${server.port}`,
                    timestamp: new Date().toISOString().replace('Z', '') + 'Z'
                });
            } else {
                socket.close();
                reject(new Error('Invalid STUN response'));
            }
        });

        socket.on('error', (err) => {
            clearTimeout(timeout);
            socket.close();
            reject(err);
        });

        // Send the STUN request
        socket.send(stunMessage, server.port, server.host, (err) => {
            if (err) {
                clearTimeout(timeout);
                socket.close();
                reject(err);
            }
        });
    });
}

// Main recording function
async function recordPublicIP(times) {
    const records = loadRecords();
    
    for (let i = 0; i < times; i++) {
        const serverIndex = i % STUN_SERVERS.length;
        const server = STUN_SERVERS[serverIndex];
        console.log(`Attempt ${i + 1}/${times} using server: ${server.host}:${server.port}`);
        
        try {
            const result = await getPublicIP(serverIndex);
            records.push(result);
            saveRecords(records);
            savePorts(records); // Save ports alongside regular records
        } catch (error) {
            console.error(`Error on attempt ${i + 1}:`, error.message);
        }
        
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    console.log(`Recorded ${times} entries. Results saved to ${RESULTS_FILE} and ${PORTS_FILE}`);
}

// Create CLI menu
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

function showMenu() {
    console.log('\n=== Public IP Recorder ===');
    console.log('1. Start Recording');
    console.log('2. View Records (Raw JSON)');
    console.log('3. View IP:Port List');
    console.log('4. View Ports Only');
    console.log('5. Exit');
    console.log('6. View Last N Ports');
    
    rl.question('Select an option: ', async (answer) => {
        switch (answer.trim()) {
            case '1':
                rl.question('How many times should we record the public IP? ', async (times) => {
                    times = parseInt(times, 10);
                    if (isNaN(times) || times <= 0) {
                        console.log('Please enter a valid number');
                        showMenu();
                        return;
                    }
                    await recordPublicIP(times);
                    showMenu();
                });
                break;
                
            case '2':
                const records = loadRecords();
                console.log('\nRecorded IPs (Raw JSON):');
                console.log(JSON.stringify(records, null, 2));
                showMenu();
                break;
                
            case '3':
                const ipRecords = loadRecords();
                console.log('\nIP:Port List with Timestamps:');
                console.log('----------------------------------------');
                ipRecords.forEach(record => {
                    console.log(`${record.ip}:${record.port} | ${record.timestamp} | via ${record.server}`);
                });
                console.log('----------------------------------------');
                console.log(`Total Records: ${ipRecords.length}`);
                showMenu();
                break;
                
            case '4':
                const portRecords = loadPorts();
                console.log('\nPort List with Timestamps:');
                console.log('----------------------------------------');
                portRecords.forEach(record => {
                    console.log(`Port: ${record.port} | ${record.timestamp}`);
                });
                console.log('----------------------------------------');
                console.log(`Total Port Records: ${portRecords.length}`);
                showMenu();
                break;
                
            case '5':
                rl.close();
                break;
                
            case '6':
                rl.question('How many recent ports do you want to see? ', (n) => {
                    const count = parseInt(n, 10);
                    if (isNaN(count) || count <= 0) {
                        console.log('Please enter a valid number');
                        showMenu();
                        return;
                    }
                    
                    const portRecords = loadPorts();
                    const lastNPorts = portRecords.slice(-count);
                    
                    console.log('\nLast ' + count + ' Port Records:');
                    console.log('----------------------------------------');
                    lastNPorts.forEach(record => {
                        console.log(`Port: ${record.port} | ${record.timestamp}`);
                    });
                    console.log('----------------------------------------');
                    console.log(`Showing ${lastNPorts.length} of ${portRecords.length} total records`);
                    showMenu();
                });
                break;
                
            default:
                console.log('Invalid option');
                showMenu();
        }
    });
}

// Start the program
showMenu();
