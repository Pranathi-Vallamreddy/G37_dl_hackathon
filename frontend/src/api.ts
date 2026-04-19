import axios from 'axios';
import type { PredictionResult, HistoryRecord, ModelInfo } from '../types';

const api = axios.create({
  baseURL: 'http://localhost:8000',
  timeout: 60000,
});

export async function predictFile(file: File): Promise<PredictionResult> {
  const form = new FormData();
  form.append('file', file);
  const { data } = await api.post<PredictionResult>('/api/predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
  return data;
}

export async function fetchHistory(limit = 50, bearingType?: number): Promise<HistoryRecord[]> {
  const params: Record<string, unknown> = { limit };
  if (bearingType !== undefined) params.bearing_type = bearingType;
  const { data } = await api.get<{ history: HistoryRecord[] }>('/api/history', { params });
  return data.history;
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const { data } = await api.get<ModelInfo>('/api/model/info');
  return data;
}

export function exportHistoryUrl(): string {
  return 'http://localhost:8000/history/export';
}
