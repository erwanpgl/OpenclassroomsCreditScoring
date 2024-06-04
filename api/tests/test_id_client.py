from modeles.predictions import predict

def test_idclient_invalid():
  try:
    print("test_idclient_invalid")
    # Arrange
    invalid_number = 4444
    # Act
    p = predict(invalid_number)
  except Exception:
     assert True

def test_client_not_accepted():
  print("test_client_not_accepted")
  # Arrange
  unsolvable_id_client = 100002
  # Act
  p = predict(unsolvable_id_client)[0]
  # Assert
  assert p == 1

def test_client_accepted():
  print("test_client_accepted")
  # Arrange
  solvable_id_client = 100004
  # Act
  p = predict(solvable_id_client)[0]
  # Assert
  assert p == 0