/**
 * API Service for communicating with the FastAPI backend
 */

import axios, { AxiosInstance } from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests if available
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle 401 errors globally
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ============================================
// TYPE DEFINITIONS
// ============================================

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
  user_info: UserInfo;
}

export interface UserInfo {
  username: string;
  role: string;
  full_name: string;
  email: string;
}

export interface PredictionRequest {
  applicant_name: string;
  credit_score: number;
  age: number;
  loan_amount: number;
  loan_term: number;
  annual_income?: number;
  loan_purpose?: string;
  sex?: string;
  race?: string;
  age_group?: string;
  zip_code?: string;
  bank_statement_text?: string;
}

export interface PredictionResponse {
  applicant_id: string;
  decision: string;
  probability: number;
  confidence: string;
  adverse_action_reasons?: string[];
  timestamp: string;
}

// ============================================
// API METHODS
// ============================================

/**
 * Authenticate user and get JWT token
 */
export const login = async (username: string, password: string): Promise<LoginResponse> => {
  const formData = new URLSearchParams();
  formData.append('username', username);
  formData.append('password', password);

  const response = await axios.post(`${API_BASE_URL}/api/token`, formData, {
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
  });

  return response.data;
};

/**
 * Submit a loan application for prediction
 */
export const submitApplication = async (
  data: PredictionRequest,
  paystub?: File,
  bankStatement?: File
): Promise<PredictionResponse> => {
  const formData = new FormData();

  // Append all form fields
  formData.append('applicant_name', data.applicant_name);
  formData.append('credit_score', data.credit_score.toString());
  formData.append('age', data.age.toString());
  formData.append('loan_amount', data.loan_amount.toString());
  formData.append('loan_term', data.loan_term.toString());

  if (data.annual_income) formData.append('annual_income', data.annual_income.toString());
  if (data.loan_purpose) formData.append('loan_purpose', data.loan_purpose);
  if (data.sex) formData.append('sex', data.sex);
  if (data.race) formData.append('race', data.race);
  if (data.age_group) formData.append('age_group', data.age_group);
  if (data.zip_code) formData.append('zip_code', data.zip_code);
  if (data.bank_statement_text) formData.append('bank_statement_text', data.bank_statement_text);

  // Append files if provided
  if (paystub) formData.append('paystub', paystub);
  if (bankStatement) formData.append('bank_statement', bankStatement);

  const response = await apiClient.post('/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

/**
 * Get explanation for a prediction
 */
export const getExplanation = async (applicant_id: string): Promise<any> => {
  const response = await apiClient.post('/explain', { applicant_id });
  return response.data;
};

/**
 * Health check
 */
export const healthCheck = async () => {
  const response = await apiClient.get('/health');
  return response.data;
};

export default apiClient;
