/**
 * Sample React Native App
 * https://github.com/facebook/react-native
 *
 * @format
 */

import React, { useState } from 'react';
import {
  SafeAreaView,
  View,
  Text,
  StyleSheet,
  Image,
  TouchableOpacity,
  ActivityIndicator,
  Alert,
  Platform,
  FlatList,
  Modal,
  ScrollView,
} from 'react-native';
import { launchCamera, launchImageLibrary, ImagePickerResponse } from 'react-native-image-picker';
import { check, request, PERMISSIONS, RESULTS } from 'react-native-permissions';
import axios from 'axios';

const API_URL = 'https://fa7e-171-251-24-72.ngrok-free.app/classify/image/mobile/detect/'; // Thay bằng URL backend thực tế

const App = () => {
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [result, setResult] = useState<string[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [modalVisible, setModalVisible] = useState(false);
  const [responseData, setResponseData] = useState<{ image_url: string; img_result: string; status: string; result: string[] } | null>(null);
  const [history, setHistory] = useState<Array<{ image_url: string; img_result: string; status: string; result: string[] }>>([]);

  const handleImagePicker = async (fromCamera: boolean) => {
    setResult(null);
    let permission;
    if (fromCamera) {
      permission = Platform.OS === 'ios' ? PERMISSIONS.IOS.CAMERA : PERMISSIONS.ANDROID.CAMERA;
    } else {
      permission =
        Platform.OS === 'ios'
          ? PERMISSIONS.IOS.PHOTO_LIBRARY
          : PERMISSIONS.ANDROID.READ_EXTERNAL_STORAGE;
    }
    const result = await check(permission);
    if (result === RESULTS.DENIED) {
      const requestResult = await request(permission);
      if (requestResult !== RESULTS.GRANTED) {
        Alert.alert('Lỗi', 'Quyền truy cập bị từ chối.');
        return;
      }
    } else if (result !== RESULTS.GRANTED) {
      Alert.alert('Lỗi', 'Quyền truy cập bị từ chối.');
      return;
    }
    const options = {
      mediaType: 'photo' as const,
      quality: 1.0,
      maxWidth: 1024,
      maxHeight: 1024,
    };
    let response: ImagePickerResponse;
    if (fromCamera) {
      response = await launchCamera(options);
    } else {
      response = await launchImageLibrary(options);
    }
    if (response.didCancel) return;
    if (response.errorCode) {
      Alert.alert('Lỗi', response.errorMessage || 'Không thể chọn ảnh');
      return;
    }
    const asset = response.assets?.[0];
    if (asset?.uri) {
      setImageUri(asset.uri);
    }
  };

  const handleUpload = async () => {
    if (!imageUri) return;
    setLoading(true);
    setResult(null);
    try {
      const fileName = imageUri.split('/').pop() || 'photo.jpg';
      const type = imageUri.endsWith('.png') ? 'image/png' : 'image/jpeg';
      const uri = Platform.OS === 'android' ? imageUri : imageUri.replace('file://', '');
      const formData = new FormData();
      formData.append('file', {
        uri,
        name: fileName,
        type,
      } as any);
      const res = await axios.post(API_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      if (res.status === 200 && res.data.status === 'success') {
        if (Array.isArray(res.data.result)) {
          setResult(res.data.result);
          const newResponseData = {
            image_url: "https://fa7e-171-251-24-72.ngrok-free.app/" + res.data.image_url,
            img_result: "https://fa7e-171-251-24-72.ngrok-free.app/" + res.data.img_result,
            status: res.data.status,
            result: res.data.result,
          };
          setResponseData(newResponseData);
          setHistory(prev => [newResponseData, ...prev]);
          setModalVisible(true);
        } else {
          Alert.alert('Lỗi', 'Kết quả từ server không đúng định dạng');
        }
      } else {
        Alert.alert('Lỗi', res.data.message || 'Không nhận được kết quả từ server');
      }
    } catch (err: any) {
      if (err.response) {
        Alert.alert('Lỗi', `Server trả về lỗi ${err.response.status}: ${err.response.data.message || 'Không xác định'}`);
      } else if (err.request) {
        Alert.alert('Lỗi', 'Không thể kết nối đến server. Kiểm tra kết nối mạng.');
      } else {
        Alert.alert('Lỗi', err.message || 'Không thể gửi ảnh');
      }
    } finally {
      setLoading(false);
    }
  };

  const renderHistoryItem = ({ item }: { item: { image_url: string; img_result: string; status: string; result: string[] } }) => (
    <TouchableOpacity
      style={styles.historyItem}
      onPress={() => {
        setResponseData(item);
        setModalVisible(true);
      }}
    >
      <Image source={{ uri: item.img_result }} style={styles.historyImage} resizeMode="cover" />
      <View style={styles.historyInfo}>
        <Text style={styles.historyStatus}>Status: {item.status}</Text>
        <Text style={styles.historyResult} numberOfLines={2}>
          {item.result.join(', ')}
        </Text>
      </View>
    </TouchableOpacity>
  );

  return (
    <SafeAreaView style={styles.container}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Text style={styles.title}>Phân loại hoa bằng ảnh</Text>
        <View style={styles.buttonRow}>
          <TouchableOpacity style={styles.button} onPress={() => handleImagePicker(false)}>
            <Text style={styles.buttonText}>Chọn ảnh</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.button} onPress={() => handleImagePicker(true)}>
            <Text style={styles.buttonText}>Chụp ảnh</Text>
          </TouchableOpacity>
        </View>
        {imageUri && (
          <View style={styles.imagePreviewContainer}>
            <Image source={{ uri: imageUri }} style={styles.imagePreview} resizeMode="contain" />
            <TouchableOpacity style={styles.uploadButton} onPress={handleUpload} disabled={loading}>
              <Text style={styles.uploadButtonText}>{loading ? 'Đang gửi...' : 'Gửi ảnh'}</Text>
            </TouchableOpacity>
          </View>
        )}
        {loading && <ActivityIndicator size="large" color="#007AFF" style={{ marginTop: 20 }} />}
        {result && (
          <View style={styles.resultContainer}>
            <Text style={styles.resultTitle}>Kết quả phân loại:</Text>
            {result.length === 0 ? (
              <Text style={styles.resultText}>Không nhận diện được hoa nào.</Text>
            ) : (
              <FlatList
                data={result}
                keyExtractor={(item, index) => index.toString()}
                renderItem={({ item }) => <Text style={styles.resultText}>{item}</Text>}
                scrollEnabled={false}
              />
            )}
          </View>
        )}
        <View style={styles.historyContainer}>
          <Text style={styles.historyTitle}>Lịch sử upload</Text>
          <FlatList
            data={history}
            keyExtractor={(item, index) => index.toString()}
            renderItem={renderHistoryItem}
            horizontal={false}
            showsVerticalScrollIndicator={false}
          />
        </View>
      </ScrollView>

      <Modal
        animationType="slide"
        transparent={false}
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.modalContainer}>
          <Text style={styles.modalTitle}>Kết quả phân loại</Text>
          <Text style={styles.modalStatus}>Status: {responseData?.status}</Text>
          <View style={styles.modalImageContainer}>
            <Text style={styles.modalImageTitle}>Ảnh kết quả:</Text>
            <Image source={{ uri: responseData?.img_result }} style={styles.modalImage} resizeMode="contain" />
          </View>
          <View style={styles.modalResultContainer}>
            <Text style={styles.modalResultTitle}>Kết quả phân loại:</Text>
            <FlatList
              data={responseData?.result}
              keyExtractor={(item, index) => index.toString()}
              renderItem={({ item }) => <Text style={styles.modalResultText}>{item}</Text>}
              scrollEnabled={false}
            />
          </View>
          <TouchableOpacity style={styles.modalCloseButton} onPress={() => setModalVisible(false)}>
            <Text style={styles.modalCloseButtonText}>Đóng</Text>
          </TouchableOpacity>
        </View>
      </Modal>
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  scrollContent: {
    padding: 24,
    alignItems: 'center',
  },
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    marginVertical: 24,
    color: '#007AFF',
  },
  buttonRow: {
    flexDirection: 'row',
    marginBottom: 24,
  },
  button: {
    backgroundColor: '#007AFF',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
    marginHorizontal: 8,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  imagePreviewContainer: {
    alignItems: 'center',
    marginBottom: 24,
  },
  imagePreview: {
    width: 250,
    height: 250,
    borderRadius: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#ccc',
  },
  uploadButton: {
    backgroundColor: '#34C759',
    paddingVertical: 10,
    paddingHorizontal: 32,
    borderRadius: 8,
  },
  uploadButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
  resultContainer: {
    marginTop: 32,
    alignItems: 'center',
  },
  resultTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 12,
    color: '#007AFF',
  },
  resultText: {
    fontSize: 18,
    marginBottom: 6,
    color: '#333',
  },
  historyContainer: {
    marginTop: 32,
    width: '100%',
  },
  historyTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 12,
    color: '#007AFF',
  },
  historyItem: {
    flexDirection: 'row',
    padding: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#ccc',
    alignItems: 'center',
  },
  historyImage: {
    width: 60,
    height: 60,
    borderRadius: 8,
  },
  historyInfo: {
    marginLeft: 12,
    flex: 1,
  },
  historyStatus: {
    fontSize: 16,
    fontWeight: '600',
    color: '#333',
  },
  historyResult: {
    fontSize: 14,
    color: '#666',
  },
  modalContainer: {
    flex: 1,
    padding: 24,
    backgroundColor: '#fff',
    alignItems: 'center',
  },
  modalTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 16,
    color: '#007AFF',
  },
  modalStatus: {
    fontSize: 18,
    marginBottom: 16,
    color: '#333',
  },
  modalImageContainer: {
    marginBottom: 16,
    alignItems: 'center',
  },
  modalImageTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
    color: '#333',
  },
  modalImage: {
    width: 300,
    height: 300,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#ccc',
  },
  modalResultContainer: {
    marginTop: 16,
    alignItems: 'center',
  },
  modalResultTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
    color: '#333',
  },
  modalResultText: {
    fontSize: 16,
    marginBottom: 4,
    color: '#333',
  },
  modalCloseButton: {
    marginTop: 24,
    backgroundColor: '#007AFF',
    paddingVertical: 12,
    paddingHorizontal: 32,
    borderRadius: 8,
  },
  modalCloseButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '600',
  },
});

export default App;