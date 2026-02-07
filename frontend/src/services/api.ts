/**
 * API service for communicating with the VDT GraphRec backend.
 * 
 * This module provides a clean interface for all API operations
 * with proper error handling and logging.
 */
import axios, { AxiosInstance, AxiosError } from 'axios';
import { Movie, RecommendationResponse, ColdStartRequest, ExplanationRequest, ExplanationResponse } from '../types';

// API configuration
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const TIMEOUT_MS = 15000;

// Custom error type for API errors
export interface ApiError {
    message: string;
    status?: number;
    detail?: any;
}

// Create custom error from axios error
function createApiError(error: AxiosError): ApiError {
    if (error.response) {
        // Server responded with error
        const detail = error.response.data as any;
        return {
            message: detail?.detail?.error || detail?.detail || detail?.message || 'Server error',
            status: error.response.status,
            detail: detail?.detail
        };
    } else if (error.request) {
        // No response received
        return {
            message: 'Network error - please check your connection',
            status: 0
        };
    } else {
        // Request setup error
        return {
            message: error.message || 'Request failed',
        };
    }
}

class ApiService {
    private client: AxiosInstance;

    constructor() {
        this.client = axios.create({
            baseURL: API_URL,
            timeout: TIMEOUT_MS,
            headers: {
                'Content-Type': 'application/json',
            },
        });

        // Request interceptor for logging
        this.client.interceptors.request.use(
            (config) => {
                console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`, config.data || '');
                return config;
            },
            (error) => {
                console.error('[API] Request error:', error);
                return Promise.reject(error);
            }
        );

        // Response interceptor for logging
        this.client.interceptors.response.use(
            (response) => {
                console.log(`[API] Response ${response.status}:`, response.data);
                return response;
            },
            (error: AxiosError) => {
                console.error('[API] Response error:', error.response?.status, error.response?.data);
                return Promise.reject(error);
            }
        );
    }

    /**
     * Get recommendations for a known user.
     */
    async getRecommendations(userId: number, topK: number = 10): Promise<{ movies: Movie[], abGroup: string }> {
        try {
            const response = await this.client.get<RecommendationResponse>(
                `/recommend/${userId}`,
                { params: { top_k: topK } }
            );
            return {
                movies: response.data.recommendations,
                abGroup: response.data.ab_group || 'control'
            };
        } catch (error) {
            console.error('[API] Error fetching recommendations:', error);
            // Return empty array for graceful degradation
            return { movies: [], abGroup: 'control' };
        }
    }

    /**
     * Get recommendations for guest/cold start users.
     * Throws ApiError on failure for proper error handling.
     */
    async getColdStartRecommendations(request: ColdStartRequest): Promise<Movie[]> {
        // Validate request before sending
        if (!request.genres?.length && !request.keywords?.length && !request.selected_movie_ids?.length && !request.query) {
            throw {
                message: 'Please provide at least one preference (genre, keyword, movie ID, or description)',
                status: 400
            } as ApiError;
        }

        try {
            const response = await this.client.post<RecommendationResponse>(
                '/recommend/cold_start',
                request
            );
            return response.data.recommendations;
        } catch (error) {
            if (axios.isAxiosError(error)) {
                const apiError = createApiError(error);
                console.error('[API] Cold start error:', apiError);
                throw apiError;
            }
            throw { message: 'Unknown error occurred' } as ApiError;
        }
    }

    /**
     * Health check for backend status.
     */
    async checkHealth(): Promise<{ healthy: boolean; usersLoaded: number }> {
        try {
            const response = await this.client.get('/health');
            return {
                healthy: response.data.status === 'healthy',
                usersLoaded: response.data.known_users || 0,
            };
        } catch {
            return { healthy: false, usersLoaded: 0 };
        }
    }

    /**
     * Explain a recommendation.
     */
    async explainRecommendation(request: ExplanationRequest): Promise<ExplanationResponse> {
        try {
            const response = await this.client.post<ExplanationResponse>(
                '/recommend/explain',
                request
            );
            return response.data;
        } catch (error) {
            console.error('[API] Explanation error:', error);
            throw createApiError(error as AxiosError);
        }
    }
}

// Export singleton instance
export const apiService = new ApiService();

// Export legacy functions for backward compatibility
export const fetchRecommendations = (userId: number) =>
    apiService.getRecommendations(userId);

export const recommendColdStart = (request: ColdStartRequest) =>
    apiService.getColdStartRecommendations(request);
