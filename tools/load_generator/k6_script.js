import http from 'k6/http';
import { sleep } from 'k6';

export const options = {
  vus: 20,
  duration: '5m',
};

export default function () {
  const payload = JSON.stringify({ payload_size: 64, cpu_hint: 0.03 });
  const headers = { 'Content-Type': 'application/json' };
  http.post('http://localhost:8001/workload', payload, { headers });
  http.get('http://localhost:8001/health');
  sleep(1);
}
