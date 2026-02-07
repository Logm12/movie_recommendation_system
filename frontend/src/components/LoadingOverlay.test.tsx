import { render, screen } from '@testing-library/react';
import { LoadingOverlay, ContentLoadingWrapper } from './LoadingOverlay';
import { MantineProvider } from '@mantine/core';

// Mock Mantine LoadingOverlay since it uses portals/animations
vi.mock('@mantine/core', async () => {
    const actual = await vi.importActual('@mantine/core');
    return {
        ...actual,
        LoadingOverlay: ({ visible }: { visible: boolean }) => (
            visible ? <div data-testid="mantine-loading-overlay">Loading...</div> : null
        ),
    };
});

describe('LoadingOverlay', () => {
    it('renders nothing when not visible', () => {
        render(
            <MantineProvider>
                <LoadingOverlay visible={false} />
            </MantineProvider>
        );
        expect(screen.queryByTestId('mantine-loading-overlay')).not.toBeInTheDocument();
    });

    it('renders overlay when visible', () => {
        render(
            <MantineProvider>
                <LoadingOverlay visible={true} />
            </MantineProvider>
        );
        expect(screen.getByTestId('mantine-loading-overlay')).toBeInTheDocument();
    });
});

describe('ContentLoadingWrapper', () => {
    it('renders children', () => {
        render(
            <MantineProvider>
                <ContentLoadingWrapper loading={false}>
                    <div data-testid="child-content">Child Content</div>
                </ContentLoadingWrapper>
            </MantineProvider>
        );
        expect(screen.getByTestId('child-content')).toHaveTextContent('Child Content');
    });

    it('shows overlay when loading is true', () => {
        render(
            <MantineProvider>
                <ContentLoadingWrapper loading={true}>
                    <div>Content</div>
                </ContentLoadingWrapper>
            </MantineProvider>
        );
        expect(screen.getByTestId('mantine-loading-overlay')).toBeInTheDocument();
    });
});
