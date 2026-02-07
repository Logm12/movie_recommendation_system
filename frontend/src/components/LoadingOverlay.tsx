/**
 * LoadingOverlay Component
 * 
 * A reusable loading overlay with animation that covers the content area
 * when data is being fetched.
 */
import { LoadingOverlay as MantineLoadingOverlay, Box } from '@mantine/core';

interface LoadingOverlayProps {
    visible: boolean;
    message?: string;
}

export function LoadingOverlay({ visible, message = 'Loading...' }: LoadingOverlayProps) {
    return (
        <MantineLoadingOverlay
            visible={visible}
            zIndex={1000}
            overlayProps={{
                radius: "sm",
                blur: 2,
                backgroundOpacity: 0.6
            }}
            loaderProps={{
                color: 'red',
                type: 'dots',
                size: 'xl'
            }}
        />
    );
}

/**
 * ContentLoadingWrapper Component
 * 
 * Wraps content with a loading overlay that shows when loading is true.
 */
interface ContentLoadingWrapperProps {
    loading: boolean;
    children: React.ReactNode;
}

export function ContentLoadingWrapper({ loading, children }: ContentLoadingWrapperProps) {
    return (
        <Box pos="relative" style={{ minHeight: 200 }}>
            <LoadingOverlay visible={loading} />
            {children}
        </Box>
    );
}

export default LoadingOverlay;
