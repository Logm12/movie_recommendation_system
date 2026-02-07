/**
 * TypeScript type definitions for the VDT GraphRec application.
 */

export interface Movie {
    id: number;
    title: string;
    genres: string;
    poster_url?: string;
    score: number;
}

export interface RecommendationResponse {
    user_id: number;
    recommendations: Movie[];
    ab_group?: string;
}

export interface ColdStartRequest {
    selected_movie_ids?: number[];
    genres?: string[];
    keywords?: string[];
    query?: string; // Neural Search
    top_k?: number;
}

/** Stores what user selected for display purposes */
export interface GuestCriteria {
    genres: string[];
    keywords: string[];
}

export interface User {
    id: string;
    label: string;
}

export const AVAILABLE_GENRES = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
    'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
    'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Thriller', 'War', 'Western'
] as const;


export interface ExplanationRequest {
    user_id: number;
    movie_id: number;
    movie_title: string;
    movie_genres: string;
}

export interface ExplanationResponse {
    explanation: string;
    method: string;
}
