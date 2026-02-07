/**
 * Notification utilities for user feedback
 * 
 * Provides functions to show success, error, and info notifications
 * using Mantine's notification system.
 */
import { notifications } from '@mantine/notifications';

export interface NotificationOptions {
    title?: string;
    message: string;
    autoClose?: number | boolean;
}

/**
 * Show an error notification
 */
export function showErrorNotification({
    title = 'Error',
    message,
    autoClose = 5000
}: NotificationOptions) {
    notifications.show({
        title: `❌ ${title}`,
        message,
        color: 'red',
        autoClose,
        withBorder: true,
        styles: {
            root: { backgroundColor: '#2C1810' },
            title: { color: '#FF6B6B' },
            description: { color: '#E0E0E0' }
        }
    });
}

/**
 * Show a success notification
 */
export function showSuccessNotification({
    title = 'Success',
    message,
    autoClose = 3000
}: NotificationOptions) {
    notifications.show({
        title: `✅ ${title}`,
        message,
        color: 'green',
        autoClose,
        withBorder: true,
        styles: {
            root: { backgroundColor: '#102C10' },
            title: { color: '#6BCB77' },
            description: { color: '#E0E0E0' }
        }
    });
}

/**
 * Show an info notification
 */
export function showInfoNotification({
    title = 'Info',
    message,
    autoClose = 4000
}: NotificationOptions) {
    notifications.show({
        title: `ℹ️ ${title}`,
        message,
        color: 'blue',
        autoClose,
        withBorder: true,
        styles: {
            root: { backgroundColor: '#10202C' },
            title: { color: '#74B9FF' },
            description: { color: '#E0E0E0' }
        }
    });
}

/**
 * Show a warning notification
 */
export function showWarningNotification({
    title = 'Warning',
    message,
    autoClose = 4000
}: NotificationOptions) {
    notifications.show({
        title: `⚠️ ${title}`,
        message,
        color: 'yellow',
        autoClose,
        withBorder: true,
        styles: {
            root: { backgroundColor: '#2C2810' },
            title: { color: '#FFC300' },
            description: { color: '#E0E0E0' }
        }
    });
}
