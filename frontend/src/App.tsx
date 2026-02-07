/**
 * VDT GraphRec Pro - Main Application Component
 * 
 * A hybrid movie recommendation system powered by LightGCN and Qdrant.
 */
import { useState } from 'react';
import {
    AppShell, Title, Group, Select, Text, SimpleGrid,
    Card, Image, Badge, Button, Container, Box, Skeleton, Modal
} from '@mantine/core';
import { motion } from 'framer-motion';

// Hooks and Services
import { useRecommendations } from './hooks';
import { Movie } from './types';

// Components
import { ColdStartModal } from './ColdStartModal';
import { MovieDetailModal } from './MovieDetailModal';
import { ContentLoadingWrapper, showInfoNotification, showErrorNotification } from './components';

// Constants
const USER_OPTIONS = [
    { value: '1', label: 'User 1' },
    { value: '10', label: 'User 10' },
    { value: '50', label: 'User 50' },
    { value: '100', label: 'User 100' },
    { value: 'guest', label: 'Guest Mode (Custom)' }
];

const HERO_BACKGROUND = 'https://images.unsplash.com/photo-1574267432553-4b4628081c31?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&q=80';

// Skeleton card component for loading state
function MovieCardSkeleton() {
    return (
        <Card shadow="sm" p="0" radius="md" style={{ backgroundColor: '#1F1F1F', height: '100%' }}>
            <Card.Section>
                <Skeleton height={300} radius={0} />
            </Card.Section>
            <div style={{ padding: '16px' }}>
                <Skeleton height={20} width="80%" mb="xs" />
                <Skeleton height={16} width="40%" mb="sm" />
                <Skeleton height={14} width="60%" />
            </div>
        </Card>
    );
}

function App() {
    // State
    const [modalOpen, setModalOpen] = useState(false);
    const [selectedMovie, setSelectedMovie] = useState<Movie | null>(null);

    // Custom hook for recommendations
    const {
        movies,
        setMovies,
        userId,
        isGuestMode,
        isLoading,
        error,
        setUserId,
        guestCriteria,
        abGroup
    } = useRecommendations({
        initialUserId: '1'
    });

    // Explanation State
    const [explanation, setExplanation] = useState<{ text: string, movie: string } | null>(null);
    const [explainingId, setExplainingId] = useState<number | null>(null);

    // Handlers
    const handleExplain = async (movie: Movie) => {
        setExplainingId(movie.id);
        try {
            // Import apiService dynamically or use from scope if available. 
            // Better to move apiService import to top-level if not already there.
            // Assuming apiService is imported.
            const { apiService } = await import('./services');
            const response = await apiService.explainRecommendation({
                user_id: userId === 'guest' ? 0 : parseInt(userId || '0'),
                movie_id: movie.id,
                movie_title: movie.title,
                movie_genres: movie.genres
            });
            setExplanation({
                text: response.explanation,
                movie: movie.title
            });
        } catch (e) {
            showErrorNotification({
                title: 'Error',
                message: 'Could not generate explanation'
            });
        } finally {
            setExplainingId(null);
        }
    };

    const handleGuestMode = () => {
        setUserId('guest');
        setModalOpen(true);
    };

    const handleUserChange = (val: string | null) => {
        if (val === 'guest') {
            handleGuestMode();
        } else if (val) {
            setUserId(val);
            showInfoNotification({
                title: 'User Changed',
                message: `Loading recommendations for User ${val}...`
            });
        }
    };

    // Helper for image URL
    const getImageUrl = (movie: Movie) =>
        movie.poster_url || `https://loremflickr.com/300/450/movie,poster?lock=${movie.id}`;

    return (
        <AppShell header={{ height: 70 }} padding="md">
            {/* Header */}
            <AppShell.Header
                p="xs"
                style={{
                    background: 'rgba(0,0,0,0.8)',
                    backdropFilter: 'blur(10px)',
                    borderBottom: 'none'
                }}
            >
                <Container
                    size="xl"
                    style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        height: '100%'
                    }}
                >
                    <Group>
                        <div style={{
                            width: 32,
                            height: 32,
                            backgroundColor: '#E50914',
                            borderRadius: '50%'
                        }} />
                        <Title order={3} style={{ color: '#fff', letterSpacing: '1px' }}>
                            Movie Recommendation System
                        </Title>
                    </Group>

                    <Group>
                        <Text fw={500} size="sm" c="dimmed">Viewing as:</Text>
                        <Select
                            placeholder="Select User"
                            data={USER_OPTIONS}
                            value={userId}
                            onChange={handleUserChange}
                            style={{ width: 180 }}
                            variant="filled"
                            disabled={isLoading}
                        />
                        {/* A/B Group Indicator */}
                        <Badge
                            variant="light"
                            color={abGroup === 'treatment' ? 'grape' : 'gray'}
                            title={`Experiment Group: ${abGroup}`}
                        >
                            {abGroup === 'treatment' ? 'Beta' : 'Stable'}
                        </Badge>
                        <Button
                            variant="outline"
                            color="gray"
                            size="xs"
                            onClick={() => setModalOpen(true)}
                            disabled={isLoading}
                        >
                            Guest Settings
                        </Button>
                    </Group>
                </Container>
            </AppShell.Header>

            {/* Main Content */}
            <AppShell.Main style={{
                backgroundColor: '#141414',
                minHeight: '100vh',
                color: '#f8f9fa'
            }}>
                {/* Hero Section */}
                <Container size="xl" mb={40}>
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5 }}
                    >
                        <div style={{
                            padding: '60px 0',
                            textAlign: 'center',
                            backgroundImage: `linear-gradient(180deg, rgba(20,20,20,0) 0%, rgba(20,20,20,1) 100%), url(${HERO_BACKGROUND})`,
                            backgroundSize: 'cover',
                            backgroundPosition: 'center',
                            borderRadius: '16px',
                            border: '1px solid rgba(255,255,255,0.1)'
                        }}>
                            <Title order={1} size={48} mb="md">
                                Hybrid AI Recommendations
                            </Title>
                            <Text size="xl" c="dimmed" mb="xl">
                                Powered by LightGCN & Qdrant Vector Search
                            </Text>
                            <Button
                                size="lg"
                                color="red"
                                radius="md"
                                onClick={handleGuestMode}
                                loading={isLoading && isGuestMode}
                            >
                                Try Guest Mode
                            </Button>
                        </div>
                    </motion.div>
                </Container>

                {/* Recommendations Grid */}
                <Container size="xl">
                    <Group justify="space-between" mb="lg">
                        <div>
                            <Title order={2}>
                                Top Picks for You {' '}
                                <Text span c="red" inherit>
                                    ({isGuestMode ? 'Guest Profile' : `User ${userId}`})
                                </Text>
                            </Title>
                            {/* Show selected criteria for guest mode */}
                            {isGuestMode && guestCriteria && guestCriteria.genres.length > 0 && (
                                <Group gap="xs" mt="xs">
                                    <Text size="sm" c="dimmed">Based on:</Text>
                                    {guestCriteria.genres.map(genre => (
                                        <Badge key={genre} color="red" variant="light" size="sm">
                                            {genre}
                                        </Badge>
                                    ))}
                                </Group>
                            )}
                        </div>
                        {isLoading && (
                            <Badge color="blue" variant="light" size="lg">
                                Loading...
                            </Badge>
                        )}
                    </Group>

                    <Box pos="relative" style={{ minHeight: 400 }}>
                        <ContentLoadingWrapper loading={isLoading}>
                            <SimpleGrid cols={{ base: 1, sm: 2, md: 5 }} spacing="lg">
                                {isLoading ? (
                                    // Show skeleton cards while loading
                                    Array.from({ length: 10 }).map((_, index) => (
                                        <MovieCardSkeleton key={`skeleton-${index}`} />
                                    ))
                                ) : (
                                    movies.map((movie, index) => (
                                        <motion.div
                                            key={movie.id}
                                            initial={{ opacity: 0, scale: 0.9 }}
                                            animate={{ opacity: 1, scale: 1 }}
                                            transition={{ delay: index * 0.05 }}
                                            whileHover={{ scale: 1.05, zIndex: 10 }}
                                            onClick={() => setSelectedMovie(movie)}
                                            style={{ cursor: 'pointer' }}
                                        >
                                            <Card
                                                shadow="sm"
                                                p="0"
                                                radius="md"
                                                style={{
                                                    backgroundColor: '#1F1F1F',
                                                    border: 'none',
                                                    height: '100%'
                                                }}
                                            >
                                                <Card.Section>
                                                    <Image
                                                        src={getImageUrl(movie)}
                                                        height={300}
                                                        alt={movie.title}
                                                        fallbackSrc="https://placehold.co/300x450?text=No+Preview"
                                                    />
                                                </Card.Section>

                                                <div style={{ padding: '16px' }}>
                                                    <Group justify="space-between" mb="xs">
                                                        <Text
                                                            fw={700}
                                                            lineClamp={1}
                                                            title={movie.title}
                                                            style={{ color: 'white' }}
                                                        >
                                                            {movie.title}
                                                        </Text>
                                                    </Group>

                                                    <Group mb="sm" gap="xs" justify="space-between">
                                                        <Group gap="xs">
                                                            {movie.score > 0 && (
                                                                <Badge color="green" variant="light" size="sm">
                                                                    {Math.round(movie.score * 100)}% Match
                                                                </Badge>
                                                            )}
                                                        </Group>
                                                        <Button
                                                            variant="subtle"
                                                            color="grape"
                                                            size="xs"
                                                            loading={explainingId === movie.id}
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                handleExplain(movie);
                                                            }}
                                                        >
                                                            Why this?
                                                        </Button>
                                                    </Group>

                                                    {/* Genre badges - highlight matching genres */}
                                                    <Group gap={4} style={{ flexWrap: 'wrap' }}>
                                                        {movie.genres.split('|').slice(0, 3).map(genre => {
                                                            const isMatch = isGuestMode && guestCriteria?.genres.some(
                                                                g => g.toLowerCase() === genre.trim().toLowerCase()
                                                            );
                                                            return (
                                                                <Badge
                                                                    key={genre}
                                                                    size="xs"
                                                                    variant={isMatch ? 'filled' : 'light'}
                                                                    color={isMatch ? 'red' : 'gray'}
                                                                >
                                                                    {genre.trim()}
                                                                </Badge>
                                                            );
                                                        })}
                                                    </Group>
                                                </div>
                                            </Card>
                                        </motion.div>
                                    ))
                                )}
                            </SimpleGrid>
                        </ContentLoadingWrapper>

                        {/* Empty State */}
                        {!isLoading && movies.length === 0 && (
                            <Box
                                ta="center"
                                py="xl"
                                style={{
                                    backgroundColor: 'rgba(255,255,255,0.05)',
                                    borderRadius: '8px'
                                }}
                            >
                                <Text size="xl" c="dimmed" mb="md">
                                    {isGuestMode
                                        ? '🎬 Select genres or movies to get recommendations.'
                                        : error
                                            ? `❌ ${error}`
                                            : '🔍 No recommendations found. Is the backend running?'}
                                </Text>
                                {isGuestMode && (
                                    <Button
                                        color="red"
                                        onClick={() => setModalOpen(true)}
                                    >
                                        Set Your Preferences
                                    </Button>
                                )}
                            </Box>
                        )}
                    </Box>
                </Container>

                {/* Modals */}
                <ColdStartModal
                    opened={modalOpen}
                    onClose={() => setModalOpen(false)}
                    onRecommendations={(recs) => {
                        setMovies(recs);
                        setUserId('guest');
                    }}
                />

                <MovieDetailModal
                    movie={selectedMovie}
                    opened={!!selectedMovie}
                    onClose={() => setSelectedMovie(null)}
                />

                {/* Explanation Modal */}
                <Modal
                    opened={!!explanation}
                    onClose={() => setExplanation(null)}
                    title={`Why "${explanation?.movie}"?`}
                    centered
                >
                    <Text size="md">{explanation?.text}</Text>
                    <Group justify="flex-end" mt="md">
                        <Button onClick={() => setExplanation(null)}>Got it</Button>
                    </Group>
                </Modal>
            </AppShell.Main>
        </AppShell>
    );
}

export default App;
