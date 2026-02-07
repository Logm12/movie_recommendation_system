/**
 * Custom hook for managing recommendations state.
 */
import { useState, useEffect, useCallback } from 'react';
import { Movie, ColdStartRequest, GuestCriteria } from '../types';
import { apiService } from '../services';

interface UseRecommendationsOptions {
    initialUserId?: string;
}

interface UseRecommendationsResult {
    movies: Movie[];
    setMovies: React.Dispatch<React.SetStateAction<Movie[]>>;
    isLoading: boolean;
    error: string | null;
    userId: string | null;
    isGuestMode: boolean;
    guestCriteria: GuestCriteria | null;
    setUserId: (id: string | null) => void;
    fetchForUser: (userId: number) => Promise<void>;
    fetchColdStart: (request: ColdStartRequest) => Promise<void>;
    abGroup: string;
}

export function useRecommendations(
    options: UseRecommendationsOptions = {}
): UseRecommendationsResult {
    const [movies, setMovies] = useState<Movie[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [userId, setUserId] = useState<string | null>(options.initialUserId || '1');
    const [guestCriteria, setGuestCriteria] = useState<GuestCriteria | null>(null);

    const [abGroup, setAbGroup] = useState<string>('control');

    const isGuestMode = userId === 'guest';

    const fetchForUser = useCallback(async (id: number) => {
        setIsLoading(true);
        setError(null);
        setGuestCriteria(null); // Clear guest criteria when switching to user
        try {
            const result = await apiService.getRecommendations(id);
            setMovies(result.movies);
            setAbGroup(result.abGroup);
        } catch (e: any) {
            setError(e?.message || 'Failed to fetch recommendations');
            setMovies([]);
            setAbGroup('control');
        } finally {
            setIsLoading(false);
        }
    }, []);

    const fetchColdStart = useCallback(async (request: ColdStartRequest) => {
        setIsLoading(true);
        setError(null);
        // Store criteria for display
        setGuestCriteria({
            genres: request.genres || [],
            keywords: request.keywords || []
        });
        try {
            const results = await apiService.getColdStartRecommendations(request);
            setMovies(results);
            setUserId('guest');
        } catch (e: any) {
            setError(e?.message || 'Failed to fetch guest recommendations');
            setMovies([]);
        } finally {
            setIsLoading(false);
        }
    }, []);

    // Auto-fetch when userId changes (for non-guest users)
    useEffect(() => {
        if (userId && userId !== 'guest') {
            fetchForUser(parseInt(userId));
        }
    }, [userId, fetchForUser]);

    return {
        movies,
        setMovies,
        isLoading,
        error,
        userId,
        isGuestMode,
        guestCriteria,
        setUserId,
        fetchForUser,
        fetchColdStart,
        abGroup,
    };
}

