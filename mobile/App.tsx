/**
 * NEXUS Mobile App
 *
 * AI-Powered Maternal-Neonatal Care Platform
 * Built with Google HAI-DEF models for the MedGemma Impact Challenge
 *
 * HAI-DEF Models:
 * - MedSigLIP: Medical image analysis
 * - HeAR: Health acoustic representations
 * - MedGemma: Clinical reasoning and synthesis
 */

import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';

// Components
import { ToastProvider } from './src/components/Toast';
import { ErrorBoundary } from './src/components/ErrorBoundary';

// Screens
import HomeScreen from './src/screens/HomeScreen';
import AnemiaScreen from './src/screens/AnemiaScreen';
import JaundiceScreen from './src/screens/JaundiceScreen';
import CryAnalysisScreen from './src/screens/CryAnalysisScreen';
import ResultsScreen from './src/screens/ResultsScreen';
import CombinedAssessmentScreen from './src/screens/CombinedAssessmentScreen';
import PregnantWomanScreen from './src/screens/PregnantWomanScreen';
import NewbornScreen from './src/screens/NewbornScreen';

// Types
export type RootStackParamList = {
  Home: undefined;
  Anemia: undefined;
  Jaundice: undefined;
  CryAnalysis: undefined;
  CombinedAssessment: undefined;
  PregnantWoman: undefined;
  Newborn: undefined;
  Results: {
    type: 'anemia' | 'jaundice' | 'cry' | 'combined' | 'pregnant' | 'newborn';
    results: Record<string, unknown>;
  };
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App(): React.JSX.Element {
  return (
    <ErrorBoundary>
      <SafeAreaProvider>
        <ToastProvider>
          <NavigationContainer>
            <Stack.Navigator
              initialRouteName="Home"
              screenOptions={{
                headerStyle: {
                  backgroundColor: '#1a73e8',
                },
                headerTintColor: '#fff',
                headerTitleStyle: {
                  fontWeight: 'bold',
                },
              }}
            >
              <Stack.Screen
                name="Home"
                component={HomeScreen}
                options={{ title: 'NEXUS' }}
              />
              <Stack.Screen
                name="Anemia"
                component={AnemiaScreen}
                options={{ title: 'Anemia Screening' }}
              />
              <Stack.Screen
                name="Jaundice"
                component={JaundiceScreen}
                options={{ title: 'Jaundice Detection' }}
              />
              <Stack.Screen
                name="CryAnalysis"
                component={CryAnalysisScreen}
                options={{ title: 'Cry Analysis' }}
              />
              <Stack.Screen
                name="CombinedAssessment"
                component={CombinedAssessmentScreen}
                options={{ title: 'Full Assessment' }}
              />
              <Stack.Screen
                name="PregnantWoman"
                component={PregnantWomanScreen}
                options={{ title: 'Maternal Assessment' }}
              />
              <Stack.Screen
                name="Newborn"
                component={NewbornScreen}
                options={{ title: 'Newborn Assessment' }}
              />
              <Stack.Screen
                name="Results"
                component={ResultsScreen}
                options={{ title: 'Results' }}
              />
            </Stack.Navigator>
          </NavigationContainer>
          <StatusBar style="light" />
        </ToastProvider>
      </SafeAreaProvider>
    </ErrorBoundary>
  );
}
