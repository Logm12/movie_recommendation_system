/**
 * ColdStartModal Component
 * 
 * Modal for guest users to set their preferences for cold-start recommendations.
 * Uses Chip group for better UX instead of MultiSelect.
 */
import { useState } from 'react';
import {
    Modal, Button, TextInput, Stack, Text, Group,
    Chip, Divider, Alert
} from '@mantine/core';
import { recommendColdStart } from './services/api';
import { Movie } from './types';
import { showErrorNotification, showSuccessNotification } from './components';

// Props interface
interface ColdStartModalProps {
    opened: boolean;
    onClose: () => void;
    onRecommendations: (movies: Movie[]) => void;
}

// Available genres for selection
const GENRES = [
    'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
    'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'
];

// Genre groups for better organization
const GENRE_GROUPS = {
    'Popular': ['Action', 'Comedy', 'Drama', 'Thriller', 'Romance'],
    'Adventure & Fantasy': ['Adventure', 'Animation', 'Fantasy', 'Sci-Fi'],
    'Specialty': ['Crime', 'Documentary', 'Horror', 'Mystery', 'War', 'Western'],
    'Classic': ['Children', 'Film-Noir', 'Musical']
};

export function ColdStartModal({ opened, onClose, onRecommendations }: ColdStartModalProps) {
    // State
    const [selectedGenres, setSelectedGenres] = useState<string[]>([]);
    const [keywords, setKeywords] = useState(''); // Kept for legacy compatibility if needed, or remove
    const [query, setQuery] = useState(''); // Neural Search
    const [movieIds, setMovieIds] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Reset state when modal opens
    const handleOpen = () => {
        setError(null);
    };

    // Validate inputs
    const validateInputs = (): boolean => {
        const hasGenres = selectedGenres.length > 0;
        const hasQuery = query.trim().length > 0;
        const hasMovieIds = movieIds.trim().length > 0;

        if (!hasGenres && !hasQuery && !hasMovieIds) {
            setError('Please select at least one genre, describe what you want, or provide movie IDs.');
            return false;
        }

        setError(null);
        return true;
    };

    // Handle form submission
    const handleSubmit = async () => {
        if (!validateInputs()) {
            return;
        }

        setLoading(true);
        setError(null);

        try {
            // Parse inputs
            const idList = movieIds
                .split(',')
                .map(s => parseInt(s.trim()))
                .filter(n => !isNaN(n) && n > 0);

            console.log('[ColdStartModal] Submitting:', {
                genres: selectedGenres,
                query: query,
                movie_ids: idList
            });

            // Call API
            const recs = await recommendColdStart({
                genres: selectedGenres,
                query: query, // Send Neural Search query
                // keywords: [], // Deprecated in UI but supported in API
                selected_movie_ids: idList,
                top_k: 10
            });
            // ... (rest is same)

            if (recs.length === 0) {
                setError('No recommendations found for your preferences. Try different options.');
                showErrorNotification({
                    title: 'No Results',
                    message: 'No movies found matching your preferences. Try selecting different genres.'
                });
                return;
            }

            // Success!
            showSuccessNotification({
                title: 'Recommendations Ready!',
                message: `Found ${recs.length} movies matching your taste.`
            });

            onRecommendations(recs);
            onClose();

            // Reset form for next time
            setSelectedGenres([]);
            setKeywords('');
            setMovieIds('');

        } catch (err: any) {
            console.error('[ColdStartModal] Error:', err);

            const errorMessage = err?.response?.data?.detail?.error
                || err?.response?.data?.detail
                || err?.message
                || 'Failed to get recommendations. Please try again.';

            setError(errorMessage);
            showErrorNotification({
                title: 'Failed to Get Recommendations',
                message: errorMessage
            });
        } finally {
            setLoading(false);
        }
    };

    // Handle genre toggle
    const handleGenreToggle = (genre: string) => {
        setSelectedGenres(prev =>
            prev.includes(genre)
                ? prev.filter(g => g !== genre)
                : [...prev, genre]
        );
        // Clear error when user makes a selection
        if (error) setError(null);
    };

    // Clear all selections
    const handleClearAll = () => {
        setSelectedGenres([]);
        setKeywords('');
        setQuery('');
        setMovieIds('');
        setError(null);
    };

    return (
        <Modal
            opened={opened}
            onClose={onClose}
            title="✨ Create Your Guest Profile"
            centered
            size="lg"
            styles={{
                header: { backgroundColor: '#1A1A1A' },
                body: { backgroundColor: '#1A1A1A' },
                title: { color: '#FFF', fontWeight: 700 }
            }}
        >
            <Stack gap="md">
                <Text size="sm" c="dimmed">
                    Tell us what you like, and our AI will generate personalized recommendations instantly.
                </Text>

                {/* How it works explanation */}
                <Alert color="blue" variant="light" title="💡 How it works">
                    Movies you see will be filtered to match your selected genres.
                    The more genres you select, the more variety you'll get!
                </Alert>

                {/* Error Alert */}
                {error && (
                    <Alert
                        color="red"
                        variant="light"
                        title="Oops!"
                        withCloseButton
                        onClose={() => setError(null)}
                    >
                        {error}
                    </Alert>
                )}

                {/* Genre Selection with Chips */}
                <Divider label="Select Your Favorite Genres" labelPosition="center" />

                <Group gap="xs">
                    {selectedGenres.length > 0 && (
                        <Button
                            size="xs"
                            variant="subtle"
                            color="gray"
                            onClick={handleClearAll}
                        >
                            Clear All
                        </Button>
                    )}
                    <Text size="xs" c="dimmed">
                        {selectedGenres.length} selected
                    </Text>
                </Group>

                <Chip.Group multiple value={selectedGenres} onChange={setSelectedGenres}>
                    <Group gap="xs" style={{ flexWrap: 'wrap' }}>
                        {GENRES.map(genre => (
                            <Chip
                                key={genre}
                                value={genre}
                                variant="filled"
                                color={selectedGenres.includes(genre) ? 'red' : 'gray'}
                                size="sm"
                                styles={{
                                    label: {
                                        cursor: 'pointer',
                                        transition: 'all 0.2s ease'
                                    }
                                }}
                            >
                                {genre}
                            </Chip>
                        ))}
                    </Group>
                </Chip.Group>

                <Divider label="Additional Options (Optional)" labelPosition="center" />

                {/* Neural Search Input */}
                <TextInput
                    label="Describe what you want to watch (Neural Search)"
                    placeholder="e.g. A cyberpunk movie with deep philosophical themes"
                    description="Our AI will understand your description and find matches"
                    value={query}
                    onChange={(event) => {
                        setQuery(event.currentTarget.value);
                        if (error) setError(null);
                    }}
                    styles={{
                        input: { backgroundColor: '#252525', borderColor: '#3A3A3A' }
                    }}
                />

                {/* Movie IDs Input (Restored) */}
                <TextInput
                    label="Favorite Movie IDs (Advanced)"
                    placeholder="e.g. 1, 296, 356"
                    description="Enter MovieLens IDs if you know specific movies you like"
                    value={movieIds}
                    onChange={(event) => {
                        setMovieIds(event.currentTarget.value);
                        if (error) setError(null);
                    }}
                    styles={{
                        input: { backgroundColor: '#252525', borderColor: '#3A3A3A' }
                    }}
                />

                {/* Action Buttons */}
                <Group justify="flex-end" mt="md">
                    <Button
                        variant="default"
                        onClick={onClose}
                        disabled={loading}
                    >
                        Cancel
                    </Button>
                    <Button
                        color="red"
                        onClick={handleSubmit}
                        loading={loading}
                        disabled={selectedGenres.length === 0 && !query.trim() && !movieIds.trim()}
                    >
                        Get Recommendations
                    </Button>
                </Group>
            </Stack>
        </Modal>
    );
}
