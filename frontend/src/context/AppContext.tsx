import React, { createContext, useContext, useState, useCallback } from 'react';
import type { PredictionResult, TabName } from '../types';

interface AppState {
  activeTab: TabName;
  setActiveTab: (t: TabName) => void;
  currentPrediction: PredictionResult | null;
  setCurrentPrediction: (p: PredictionResult | null) => void;
  recentPredictions: PredictionResult[];
  addPrediction: (p: PredictionResult) => void;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [activeTab, setActiveTab] = useState<TabName>('dashboard');
  const [currentPrediction, setCurrentPrediction] = useState<PredictionResult | null>(null);
  const [recentPredictions, setRecentPredictions] = useState<PredictionResult[]>([]);

  const addPrediction = useCallback((p: PredictionResult) => {
    setCurrentPrediction(p);
    setRecentPredictions(prev => [p, ...prev].slice(0, 20));
  }, []);

  return (
    <AppContext.Provider value={{
      activeTab, setActiveTab,
      currentPrediction, setCurrentPrediction,
      recentPredictions, addPrediction,
    }}>
      {children}
    </AppContext.Provider>
  );
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useApp must be used within AppProvider');
  return ctx;
}
