import { Modal, Text, Image, Group, Badge, Stack, Button, Flex } from '@mantine/core';
import { Movie } from './api';

interface MovieDetailModalProps {
    movie: Movie | null;
    opened: boolean;
    onClose: () => void;
}

export function MovieDetailModal({ movie, opened, onClose }: MovieDetailModalProps) {
    if (!movie) return null;

    // Mock description generator since DB doesn't have plot summaries
    const getMockDescription = (m: Movie) => {
        return `Experience the thrill of "${m.title}", a masterpiece in the ${m.genres.split('|').join(' and ')} genre. 
        This film has captivated audiences with its compelling storytelling and memorable characters. 
        A must-watch for fans of ${m.genres.split('|')[0]} looking for a top-tier cinematic journey.`;
    };

    // Use LoremFlickr for distinct pseudo-random images based on ID
    // or fallback to a reliable placeholder service
    const imageUrl = movie.poster_url || `https://loremflickr.com/400/600/movie,poster?lock=${movie.id}`;

    return (
        <Modal
            opened={opened}
            onClose={onClose}
            size="lg"
            centered
            padding={0}
            styles={{ body: { padding: 0 } }}
        >
            <Flex direction={{ base: 'column', sm: 'row' }}>
                <Image
                    src={imageUrl}
                    w={{ base: '100%', sm: 300 }}
                    h={{ base: 450, sm: 'auto' }}
                    fit="cover"
                    fallbackSrc="https://placehold.co/300x450?text=No+Image"
                />
                <Stack p="xl" style={{ flex: 1 }}>
                    <Text size="xs" tt="uppercase" c="dimmed" fw={700}>Movie Details</Text>

                    <Text size="h2" fw={900} lh={1.1}>
                        {movie.title}
                    </Text>

                    <Group gap="xs">
                        {movie.score > 0 && (
                            <Badge color="green" size="lg" variant="filled">
                                {Math.round(movie.score * 100)}% Match
                            </Badge>
                        )}
                        {movie.genres.split('|').map(g => (
                            <Badge key={g} variant="outline" color="gray">{g}</Badge>
                        ))}
                    </Group>

                    <Text size="sm" mt="md" opacity={0.8} style={{ lineHeight: 1.6 }}>
                        {getMockDescription(movie)}
                    </Text>

                    <Group mt="auto">
                        <Button variant="light" color="red" fullWidth onClick={onClose}>
                            Close
                        </Button>
                    </Group>
                </Stack>
            </Flex>
        </Modal>
    );
}
